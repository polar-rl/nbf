import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.particle import ParticleFilter, ParticleSet


@eqx.filter_jit
def _nbf_density_step(model, embedding, env, x_true: Array, x_neg: Array):
    pts = jnp.concatenate([x_true[None], x_neg], axis=0)  # (M+1, *state_shape)
    pts_norm = env.normalize_flat(pts)
    logp_out = model.log_prob(pts_norm, embedding)
    logps = logp_out[0] if isinstance(logp_out, tuple) else logp_out
    return logps[0], logps[1:]


@eqx.filter_jit
def _pf_density_step(env, particle_set, x_true: Array, x_neg: Array, bandwidth: float):
    parts = env.state_array(particle_set.particles)  # (P, *state_shape)
    parts_flat = parts.reshape(parts.shape[0], -1)
    log_w = jnp.log(particle_set.weights / (jnp.sum(particle_set.weights) + 1e-30) + 1e-30)
    h2 = bandwidth * bandwidth

    def kde_logp(x):  # x: (*state_shape,)
        diff = parts_flat - x.reshape(-1)[None]  # (P, D)
        sq = jnp.sum(diff * diff, axis=-1)
        return jax.scipy.special.logsumexp(log_w - 0.5 * sq / h2)

    pts = jnp.concatenate([x_true[None], x_neg], axis=0)
    logps = jax.vmap(kde_logp)(pts)
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
            return pf_set, env.state_array(samples)

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
            samps = env.state_array(pf.sample(pf_set, state_t, num_ref_samples, samp_k))
            lp_pos, lp_neg = _pf_density_step(env, pf_set, env.state_array(state_t), neg, bandwidth)
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


@eqx.filter_jit
def rejuvenate_particles(
    pf_set: ParticleSet,
    gold_obs,
    env,
    num_steps: int,
    step_size: float,
    key: Array,
) -> ParticleSet:
    has_obs = env.observation_present(gold_obs)
    positions = env.state_array(pf_set.particles)  # (N, *state_shape)

    initial_log = env.observation_log_prob(positions, gold_obs)

    def _one_iter(carry, key_iter):
        pos, cur_log = carry
        prop_key, acc_key = jax.random.split(key_iter)
        noise = jax.random.normal(prop_key, pos.shape) * step_size
        prop = env.constrain(pos + noise)
        prop_log = env.observation_log_prob(prop, gold_obs)
        log_alpha = prop_log - cur_log
        u = jax.random.uniform(acc_key, log_alpha.shape)
        accept = jnp.log(u + 1e-30) < log_alpha
        accept_b = accept.reshape(accept.shape + (1,) * (pos.ndim - 1))
        new_pos = jnp.where(accept_b, prop, pos)
        new_log = jnp.where(accept, prop_log, cur_log)
        return (new_pos, new_log), None

    def _do_rejuvenate():
        keys = jax.random.split(key, num_steps)
        (final_pos, _), _ = jax.lax.scan(_one_iter, (positions, initial_log), keys)
        return final_pos

    final_pos = jax.lax.cond(has_obs, _do_rejuvenate, lambda: positions)
    new_state = env.replace_state_array(pf_set.particles, final_pos)
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
            pf_set = rejuvenate_particles(
                pf_set, next_obs_t, env, mcmc_steps, mcmc_step_size, mcmc_k
            )
            samples = pf.sample(pf_set, state_t, num_ref_samples, samp_k)
            return pf_set, env.state_array(samples)

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
            return embedding, env.state_array(samples)

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
            samps = env.state_array(nbf.sample(embedding, state_t, num_ref_samples, samp_k))
            lp_pos, lp_neg = _nbf_density_step(
                embedding_model, embedding, env, env.state_array(state_t), neg
            )
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
