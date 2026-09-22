import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from nbf.filters.likelihood import likelihood_features
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.particle import ParticleFilter, ParticleSet


@eqx.filter_jit
def _nbf_density_step(model, embedding, env, x_true: Array) -> Array:
    """Log-density of the model's belief at the true state, in raw state units.

    The generator's density lives in the model's normalized coordinates; the
    change-of-variables term `env.normalize_log_jacobian()` converts it so NLLs are
    comparable with methods whose densities are defined on the raw state (EnKF, GMF).
    """
    pts_norm = env.normalize_flat(x_true[None])
    logp_out = model.log_prob(pts_norm, embedding)
    logps = logp_out[0] if isinstance(logp_out, tuple) else logp_out
    return logps[0] + env.normalize_log_jacobian()


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
            pf_set, _, _, _ = pf.update(env, pf_set, state_t, prev_obs_t, 0, next_obs_t, sk)
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
            embedding, _, _, _ = nbf.update(env, embedding, state_t, prev_obs_t, 0, next_obs_t, sk)
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
    num_ref_samples: int,
):
    """Per step: `num_ref_samples` draws from the belief and the log-density at the truth."""

    def _one(state0, obs0, states_next, obs_prev, obs_next, rkey, step_keys, sample_keys):
        embedding = nbf.reset(env, state0, obs0, rkey)

        def _step(embedding, xs):
            sk, samp_k, state_t, prev_obs_t, next_obs_t = xs
            embedding, _, _, _ = nbf.update(env, embedding, state_t, prev_obs_t, 0, next_obs_t, sk)
            samps = env.state_array(nbf.sample(embedding, state_t, num_ref_samples, samp_k))
            lp_true = _nbf_density_step(embedding_model, embedding, env, env.state_array(state_t))
            return embedding, (samps, lp_true)

        _, (samps, lp_true) = jax.lax.scan(
            _step, embedding, (step_keys, sample_keys, states_next, obs_prev, obs_next)
        )
        return samps, lp_true

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
def gaussian_filter_samples_density_stream_batch(
    env,
    filt,
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
    def _gauss_logp(pset, x_true):
        X = env.state_array(pset.particles)
        X = X.reshape(X.shape[0], -1)
        x = x_true.reshape(-1)
        mu = jnp.mean(X, axis=0)
        A = X - mu
        cov = A.T @ A / (X.shape[0] - 1) + 1e-4 * jnp.eye(X.shape[1])
        chol = jnp.linalg.cholesky(cov)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        sol = jax.scipy.linalg.solve_triangular(chol, x - mu, lower=True)
        return -0.5 * (jnp.sum(sol * sol) + log_det + x.shape[0] * jnp.log(2.0 * jnp.pi))

    def _one(state0, obs0, states_next, obs_prev, obs_next, rkey, step_keys, sample_keys):
        pset = filt.reset(env, state0, obs0, rkey)

        def _step(pset, xs):
            sk, samp_k, state_t, prev_obs_t, next_obs_t = xs
            pset, _, _, _ = filt.update(env, pset, state_t, prev_obs_t, 0, next_obs_t, sk)
            samps = env.state_array(filt.sample(pset, state_t, num_ref_samples, samp_k))
            lp_true = _gauss_logp(pset, env.state_array(state_t))
            return pset, (samps, lp_true)

        _, (samps, lp_true) = jax.lax.scan(
            _step, pset, (step_keys, sample_keys, states_next, obs_prev, obs_next)
        )
        return samps, lp_true

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


def _weighted_draw(particle_set, state, num_samples: int, key: Array):
    """Multinomial draw under the set's importance weights (the SIR posterior sample);
    uniform if the weights underflowed to zero."""
    w = particle_set.weights
    total = jnp.sum(w)
    n = w.shape[0]
    probs = jnp.where(total > 0.0, w / jnp.where(total > 0.0, total, 1.0), 1.0 / n)
    indices = jax.random.choice(key, n, (num_samples,), p=probs)
    return jax.tree.map(lambda x: x[indices], particle_set.particles)


def _ess_fraction(particle_set) -> Array:
    return particle_set.neff() / particle_set.weights.shape[0]


@eqx.filter_jit
def pf_crossover_stream_batch(
    env,
    pf: ParticleFilter,
    nbf: NeuralBayesianFilter,
    density_model,
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
            sim_k, res_k = jax.random.split(sk)
            pf_k, nbf_k = jax.random.split(samp_k)
            propagated, _ = pf_set.simulate(env, 0, next_obs_t, sim_k)
            ess = _ess_fraction(propagated)
            pf_samples = env.state_array(_weighted_draw(propagated, state_t, num_ref_samples, pf_k))
            features = likelihood_features(env, propagated.particles, next_obs_t)
            embedding = nbf.embed_particles(
                ParticleSet(propagated.particles, propagated.weights, features)
            )
            regen = env.state_array(nbf.sample(embedding, state_t, num_ref_samples, nbf_k))
            lp_true = _nbf_density_step(density_model, embedding, env, env.state_array(state_t))
            pf_set = jax.lax.cond(
                propagated.neff() < pf.num_particles * pf.resample_factor,
                lambda: propagated.systematic_resample(res_k),
                lambda: propagated,
            )
            return pf_set, (pf_samples, regen, lp_true, ess)

        _, out = jax.lax.scan(
            _step, pf_set, (step_keys, sample_keys, states_next, obs_prev, obs_next)
        )
        return out

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
def nbf_crossover_stream_batch(
    env,
    nbf: NeuralBayesianFilter,
    density_model,
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
            nbf_k, pf_k = jax.random.split(samp_k)
            propagated, _, _, _ = nbf.update_particles(
                env, embedding, state_t, prev_obs_t, 0, next_obs_t, sk
            )
            ess = _ess_fraction(propagated)
            pf_samples = env.state_array(_weighted_draw(propagated, state_t, num_ref_samples, pf_k))
            embedding = nbf.embed_particles(propagated)
            regen = env.state_array(nbf.sample(embedding, state_t, num_ref_samples, nbf_k))
            lp_true = _nbf_density_step(density_model, embedding, env, env.state_array(state_t))
            return embedding, (regen, pf_samples, lp_true, ess)

        _, out = jax.lax.scan(
            _step, embedding, (step_keys, sample_keys, states_next, obs_prev, obs_next)
        )
        return out

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
