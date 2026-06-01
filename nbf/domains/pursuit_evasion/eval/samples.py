import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from nbf.filters.gmf import GMF
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.particle import ParticleFilter, ParticleSet

from nbf.domains.pursuit_evasion.env import (
    SENSOR_POS,
    PursuitEvasionEnv,
    clip_to_valid,
    normalize_positions,
)


@eqx.filter_jit
def _nbf_density_step(model, embedding, x_true: Array, x_neg: Array):
    M = x_neg.shape[0]
    pts = jnp.concatenate([x_true[None], x_neg], axis=0)  # (M+1, k, 2)
    pts_norm = normalize_positions(pts).reshape(M + 1, -1)
    logp_out = model.log_prob(pts_norm, embedding)
    logps = logp_out[0] if isinstance(logp_out, tuple) else logp_out
    return logps[0], logps[1:]


@eqx.filter_jit
def _pf_density_step(particle_set, x_true: Array, x_neg: Array, bandwidth: float):
    parts = particle_set.particles.positions  # (P, k, 2)
    log_w = jnp.log(particle_set.weights / (jnp.sum(particle_set.weights) + 1e-30) + 1e-30)
    h2 = bandwidth * bandwidth

    def kde_logp(x):  # x: (k, 2)
        diff = parts - x[None]  # (P, k, 2)
        sq = jnp.sum(diff * diff, axis=(1, 2))
        return jax.scipy.special.logsumexp(log_w - 0.5 * sq / h2)

    pts = jnp.concatenate([x_true[None], x_neg], axis=0)
    logps = jax.vmap(kde_logp)(pts)
    return logps[0], logps[1:]


@eqx.filter_jit
def _gmf_logp_step(weights, mus, sigmas, x_true: Array, x_neg: Array):
    pts = jnp.concatenate([x_true[None], x_neg], axis=0)
    pts_flat = pts.reshape(pts.shape[0], -1)  # (M+1, 2k)
    n = pts_flat.shape[-1]
    safe_w = weights / (jnp.sum(weights) + 1e-30)
    log_w = jnp.log(safe_w + 1e-30)

    def comp_logpdf(mu, sig, x):
        sig_sym = 0.5 * (sig + sig.T) + 1e-6 * jnp.eye(n)
        L = jnp.linalg.cholesky(sig_sym)
        diff = x - mu
        sol = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return -0.5 * (jnp.sum(sol * sol) + log_det + n * jnp.log(2.0 * jnp.pi))

    def logpdf(x):
        per_comp = jax.vmap(lambda mu, sig: comp_logpdf(mu, sig, x))(mus, sigmas)
        return jax.scipy.special.logsumexp(log_w + per_comp)

    logps = jax.vmap(logpdf)(pts_flat)
    return logps[0], logps[1:]


@eqx.filter_jit
def pf_reference_sample_stream_batch(
    env,
    pf: ParticleFilter,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    reset_keys: Array,
    step_keys_b: Array,
    sample_keys_b: Array,
    num_ref_samples: int,
):

    def _one(state0, obs0, states_next, obs_prev, obs_next, rkey, step_keys, sample_keys):
        pf_set = pf.reset(env, state0, obs0, rkey)

        def _step(pf_set, xs):
            sk, samp_k, state_t, prev_obs_t, next_obs_t = xs
            pf_set, _ = pf.update(env, pf_set, state_t, prev_obs_t, 0, next_obs_t, sk)
            samples = pf.sample(pf_set, state_t, num_ref_samples, samp_k)
            return pf_set, samples.positions

        _, samps = jax.lax.scan(
            _step, pf_set, (step_keys, sample_keys, states_next, obs_prev, obs_next)
        )
        return samps

    return jax.vmap(_one)(
        state0_b,
        obs0_b,
        states_next_b,
        obs_prev_b,
        obs_next_b,
        reset_keys,
        step_keys_b,
        sample_keys_b,
    )


@eqx.filter_jit
def pf_samples_density_stream_batch(
    env,
    pf: ParticleFilter,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    reset_keys: Array,
    step_keys_b: Array,
    sample_keys_b: Array,
    neg_pts_b: Array,
    num_ref_samples: int,
    bandwidth: float,
):
    def _one(state0, obs0, states_next, obs_prev, obs_next, rkey, step_keys, sample_keys, neg_pts):
        pf_set = pf.reset(env, state0, obs0, rkey)

        def _step(pf_set, xs):
            sk, samp_k, state_t, prev_obs_t, next_obs_t, neg = xs
            pf_set, _ = pf.update(env, pf_set, state_t, prev_obs_t, 0, next_obs_t, sk)
            samps = pf.sample(pf_set, state_t, num_ref_samples, samp_k).positions
            lp_pos, lp_neg = _pf_density_step(pf_set, state_t.positions, neg, bandwidth)
            return pf_set, (samps, lp_pos, lp_neg)

        _, (samps, lp_pos, lp_neg) = jax.lax.scan(
            _step,
            pf_set,
            (step_keys, sample_keys, states_next, obs_prev, obs_next, neg_pts),
        )
        return samps, lp_pos, lp_neg

    return jax.vmap(_one)(
        state0_b,
        obs0_b,
        states_next_b,
        obs_prev_b,
        obs_next_b,
        reset_keys,
        step_keys_b,
        sample_keys_b,
        neg_pts_b,
    )


def _obs_log_likelihood(positions: Array, gold_obs, env: PursuitEvasionEnv) -> Array:
    sigma = gold_obs.observation_sigma
    pred_ranges = jnp.linalg.norm(positions - SENSOR_POS, axis=-1)  # (N, k)
    if env.shuffle_observations and env.num_evaders > 1:
        permuted = gold_obs.ranges[env.perm_indices]  # (P, k)
        diff = pred_ranges[:, None, :] - permuted[None, :, :]
        log_per_perm = -0.5 * jnp.sum((diff / sigma) ** 2, axis=-1)
        return jax.scipy.special.logsumexp(log_per_perm, axis=1)
    return -0.5 * jnp.sum(((pred_ranges - gold_obs.ranges) / sigma) ** 2, axis=-1)


@eqx.filter_jit
def _rejuvenate_positions(
    pf_set: ParticleSet,
    gold_obs,
    env: PursuitEvasionEnv,
    num_steps: int,
    step_size: float,
    key: Array,
) -> ParticleSet:
    has_obs = gold_obs.ranges[0] >= 0.0
    obstacles = env.obstacles
    positions = pf_set.particles.positions  # (N, k, 2)

    initial_log = _obs_log_likelihood(positions, gold_obs, env)

    def _one_iter(carry, key_iter):
        pos, cur_log = carry
        prop_key, acc_key = jax.random.split(key_iter)
        noise = jax.random.normal(prop_key, pos.shape) * step_size
        prop = pos + noise
        prop = jax.vmap(jax.vmap(lambda p: clip_to_valid(p, obstacles)))(prop)
        prop_log = _obs_log_likelihood(prop, gold_obs, env)
        log_alpha = prop_log - cur_log
        u = jax.random.uniform(acc_key, log_alpha.shape)
        accept = jnp.log(u + 1e-30) < log_alpha
        new_pos = jnp.where(accept[:, None, None], prop, pos)
        new_log = jnp.where(accept, prop_log, cur_log)
        return (new_pos, new_log), None

    def _do_rejuvenate():
        keys = jax.random.split(key, num_steps)
        (final_pos, _), _ = jax.lax.scan(_one_iter, (positions, initial_log), keys)
        return final_pos

    final_pos = jax.lax.cond(has_obs, _do_rejuvenate, lambda: positions)
    new_state = eqx.tree_at(lambda s: s.positions, pf_set.particles, final_pos)
    return ParticleSet(new_state, pf_set.weights)


@eqx.filter_jit
def gold_smc_reference_sample_stream_batch(
    env,
    pf: ParticleFilter,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    reset_keys: Array,
    step_keys_b: Array,
    rejuvenate_keys_b: Array,
    sample_keys_b: Array,
    num_ref_samples: int,
    mcmc_steps: int,
    mcmc_step_size: float,
):
    def _one(state0, obs0, states_next, obs_prev, obs_next, rkey, step_keys, rej_keys, samp_keys):
        pf_set = pf.reset(env, state0, obs0, rkey)

        def _step(pf_set, xs):
            sk, rk, samp_k, state_t, prev_obs_t, next_obs_t = xs
            pf_set, _ = pf.update(env, pf_set, state_t, prev_obs_t, 0, next_obs_t, sk)
            force_rk, mcmc_k = jax.random.split(rk)
            pf_set = pf_set.systematic_resample(force_rk)
            pf_set = _rejuvenate_positions(
                pf_set, next_obs_t, env, mcmc_steps, mcmc_step_size, mcmc_k
            )
            samples = pf.sample(pf_set, state_t, num_ref_samples, samp_k)
            return pf_set, samples.positions

        _, samps = jax.lax.scan(
            _step, pf_set, (step_keys, rej_keys, samp_keys, states_next, obs_prev, obs_next)
        )
        return samps

    return jax.vmap(_one)(
        state0_b,
        obs0_b,
        states_next_b,
        obs_prev_b,
        obs_next_b,
        reset_keys,
        step_keys_b,
        rejuvenate_keys_b,
        sample_keys_b,
    )


@eqx.filter_jit
def nbf_reference_sample_stream_batch(
    env,
    nbf: NeuralBayesianFilter,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    reset_keys: Array,
    step_keys_b: Array,
    sample_keys_b: Array,
    num_ref_samples: int,
):

    def _one(state0, obs0, states_next, obs_prev, obs_next, rkey, step_keys, sample_keys):
        embedding = nbf.reset(env, state0, obs0, rkey)

        def _step(embedding, xs):
            sk, samp_k, state_t, prev_obs_t, next_obs_t = xs
            embedding, _ = nbf.update(env, embedding, state_t, prev_obs_t, 0, next_obs_t, sk)
            samples = nbf.sample(embedding, state_t, num_ref_samples, samp_k)
            return embedding, samples.positions

        _, samps = jax.lax.scan(
            _step,
            embedding,
            (step_keys, sample_keys, states_next, obs_prev, obs_next),
        )
        return samps

    return jax.vmap(_one)(
        state0_b,
        obs0_b,
        states_next_b,
        obs_prev_b,
        obs_next_b,
        reset_keys,
        step_keys_b,
        sample_keys_b,
    )


@eqx.filter_jit
def nbf_samples_density_stream_batch(
    env,
    nbf: NeuralBayesianFilter,
    embedding_model,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    reset_keys: Array,
    step_keys_b: Array,
    sample_keys_b: Array,
    neg_pts_b: Array,
    num_ref_samples: int,
):
    def _one(state0, obs0, states_next, obs_prev, obs_next, rkey, step_keys, sample_keys, neg_pts):
        embedding = nbf.reset(env, state0, obs0, rkey)

        def _step(embedding, xs):
            sk, samp_k, state_t, prev_obs_t, next_obs_t, neg = xs
            embedding, _ = nbf.update(env, embedding, state_t, prev_obs_t, 0, next_obs_t, sk)
            samps = nbf.sample(embedding, state_t, num_ref_samples, samp_k).positions
            lp_pos, lp_neg = _nbf_density_step(embedding_model, embedding, state_t.positions, neg)
            return embedding, (samps, lp_pos, lp_neg)

        _, (samps, lp_pos, lp_neg) = jax.lax.scan(
            _step,
            embedding,
            (step_keys, sample_keys, states_next, obs_prev, obs_next, neg_pts),
        )
        return samps, lp_pos, lp_neg

    return jax.vmap(_one)(
        state0_b,
        obs0_b,
        states_next_b,
        obs_prev_b,
        obs_next_b,
        reset_keys,
        step_keys_b,
        sample_keys_b,
        neg_pts_b,
    )


@eqx.filter_jit
def _gmf_sample_mixture(weights, mus, sigmas, S: int, k: int, key: Array) -> Array:
    pick_key, noise_key = jax.random.split(key)
    safe_w = weights / (jnp.sum(weights) + 1e-30)
    idx = jax.random.choice(pick_key, mus.shape[0], (S,), p=safe_w)
    mu_sel = mus[idx]
    sig_sel = sigmas[idx]
    n = mu_sel.shape[-1]
    # Cholesky with jitter for numerical stability under near-degenerate modes
    jitter = 1e-6 * jnp.eye(n)
    L_chol = jnp.linalg.cholesky(sig_sel + jitter[None])
    eps = jax.random.normal(noise_key, (S, n))
    draws = mu_sel + jnp.einsum('sij,sj->si', L_chol, eps)
    return draws.reshape(S, k, 2)


@eqx.filter_jit
def gmf_reference_sample_stream_batch(
    gmf: GMF,
    obstacles,
    policy_tuple,
    states_next_b,
    obs_next_b,
    init_keys: Array,
    sample_keys_b: Array,
    num_ref_samples: int,
):
    k = gmf.ukf.k

    def _one(states_next, obs_next, ikey, sample_keys):
        w0, mu0, sg0 = gmf.init(key=ikey)

        def _step(carry, xs):
            w, mu, sg = carry
            samp_k, _state_t, next_obs_t = xs
            w, mu, sg = gmf.step(w, mu, sg, policy_tuple, obstacles, next_obs_t)
            samps = _gmf_sample_mixture(w, mu, sg, num_ref_samples, k, samp_k)
            return (w, mu, sg), samps

        _, samps = jax.lax.scan(_step, (w0, mu0, sg0), (sample_keys, states_next, obs_next))
        return samps

    return jax.vmap(_one)(
        states_next_b,
        obs_next_b,
        init_keys,
        sample_keys_b,
    )


@eqx.filter_jit
def gmf_samples_density_stream_batch(
    env,
    gmf: GMF,
    states_next_b,
    obs_next_b,
    init_keys: Array,
    sample_keys_b: Array,
    neg_pts_b: Array,
    num_ref_samples: int,
):
    k = gmf.ukf.k

    def _one(states_next, obs_next, ikey, sample_keys, neg_pts):
        w0, mu0, sg0 = gmf.init(key=ikey)

        def _step(carry, xs):
            w, mu, sg = carry
            samp_k, state_t, next_obs_t, neg = xs
            w, mu, sg = gmf.step(w, mu, sg, env.policy, env.obstacles, next_obs_t)
            samps = _gmf_sample_mixture(w, mu, sg, num_ref_samples, k, samp_k)
            lp_pos, lp_neg = _gmf_logp_step(w, mu, sg, state_t.positions, neg)
            return (w, mu, sg), (samps, lp_pos, lp_neg)

        _, (samps, lp_pos, lp_neg) = jax.lax.scan(
            _step, (w0, mu0, sg0), (sample_keys, states_next, obs_next, neg_pts)
        )
        return samps, lp_pos, lp_neg

    return jax.vmap(_one)(states_next_b, obs_next_b, init_keys, sample_keys_b, neg_pts_b)
