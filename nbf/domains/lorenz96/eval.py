import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from nbf.domains.lorenz96.env import (
    Lorenz96Env,
    conditional_gaussian_redraw,
    rk4_window,
)
from nbf.domains.lorenz96.generate_data import systematic_indices
from nbf.eval.base import batch_size, split_T, time_len


@eqx.filter_jit
def gold_smc_sample_stream_batch(
    env: Lorenz96Env,
    state0_b,
    obs_values_b: Array,
    prior_keys: Array,
    step_keys_b: Array,
    num_particles: int,
    num_ref_samples: int,
):
    obs_idx = jnp.arange(1, env.K, 2)

    def _one(state0, obs_values, prior_key, step_keys):
        forcing, sigma_obs = state0.forcing, state0.sigma_obs
        parts = state0.center + env.sigma_init * jax.random.normal(
            prior_key, (num_particles, env.K)
        )

        def _step(parts, xs):
            k, y = xs
            prop_key, res_key, rej_key, samp_key = jax.random.split(k, 4)
            means = rk4_window(parts, forcing, env.dt, env.substeps, env.linear)
            parts = means + env.sigma_dyn * jax.random.normal(prop_key, parts.shape)
            log_w = -0.5 * jnp.sum(((parts[:, obs_idx] - y) / sigma_obs) ** 2, axis=-1)
            w = jax.nn.softmax(log_w)
            idx = systematic_indices(w, res_key)
            parts = conditional_gaussian_redraw(means[idx], y, env.sigma_dyn, sigma_obs, rej_key)
            samp_idx = jax.random.randint(samp_key, (num_ref_samples,), 0, num_particles)
            return parts, parts[samp_idx]

        _, samps = jax.lax.scan(_step, parts, (step_keys, obs_values))
        return samps

    return jax.vmap(_one)(state0_b, obs_values_b, prior_keys, step_keys_b)


def compute_l96_gold_smc_samples(
    env: Lorenz96Env,
    pf_count: int,
    num_ref_samples: int,
    state0_b,
    obs_next_b,
    key: Array,
):
    N = batch_size(state0_b)
    T = time_len(obs_next_b)
    prior_key, step_key = jax.random.split(key)
    return gold_smc_sample_stream_batch(
        env,
        state0_b,
        obs_next_b.values,
        jax.random.split(prior_key, N),
        split_T(step_key, N, T),
        pf_count,
        num_ref_samples,
    )
