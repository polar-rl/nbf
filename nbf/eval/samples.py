import equinox as eqx
import jax
from jax import Array

from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.particle import ParticleFilter


@eqx.filter_jit
def _nbf_density_step(model, embedding, env, x_true: Array) -> Array:
    """Log-density of the model's belief at the true state (the reference-free score)."""
    pts_norm = env.normalize_flat(x_true[None])
    logp_out = model.log_prob(pts_norm, embedding)
    logps = logp_out[0] if isinstance(logp_out, tuple) else logp_out
    return logps[0]


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
