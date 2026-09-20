import equinox as eqx
import jax
import jax.numpy as jnp

from nbf.domains.pursuit_evasion.env import (
    SENSOR_POS,
    SIGMA_DYN,
    SIGMA_OBS_MAX,
    L,
    PursuitEvasionEnv,
    PursuitEvasionObservation,
    propagate_deterministic,
)
from nbf.filters.gmf import GMF


def make_pe_gmf(
    num_evaders: int,
    M: int = 1,
    observation_sigma_scale: float = 1.0,
    kl_merge: float = 0.5,
    weight_prune: float = 0.01,
) -> GMF:
    n = 2 * num_evaders
    return GMF(
        num_evaders=num_evaders,
        sigma_dyn=SIGMA_DYN,
        sigma_obs=SIGMA_OBS_MAX,
        sensor_pos=SENSOR_POS,
        propagate_fn=propagate_deterministic,
        init_mu=jnp.full(n, L / 2),
        init_sigma=((L / 2) ** 2 * jnp.eye(n)),
        M=M,
        jitter_scale=0.1 * L,
        kl_merge=kl_merge,
        weight_prune=weight_prune,
        observation_sigma_scale=observation_sigma_scale,
    )


def make_env_kn(
    k: int, episode_length: int, policy_tuple, num_sensors: int = 1, obs_interval: int = 1
) -> PursuitEvasionEnv:
    return PursuitEvasionEnv(
        num_evaders=k,
        time_limit=episode_length,
        policy=policy_tuple,
        shuffle_observations=True,
        num_sensors=num_sensors,
        obs_interval=obs_interval,
    )


def override_sigma(env, state0_b, obs0_b, states_next_b, sigma: float, key):
    """Re-noise a rolled-out batch at a pinned observation sigma.

    Positions do not depend on the observation noise, so one rollout serves every
    sigma of a sweep; the ranges are redrawn from the true positions at `sigma` and the
    sigma fields of states and observations are overwritten so the filters weight with
    the pinned value. Ranges come back in canonical evader order; with shuffled
    observations the likelihood is permutation-marginalized, so the order is immaterial.
    """
    sigma_arr = jnp.asarray(sigma, dtype=jnp.float32)
    state0_b = eqx.tree_at(
        lambda s: s.observation_sigma,
        state0_b,
        jnp.broadcast_to(sigma_arr, state0_b.observation_sigma.shape),
    )
    states_next_b = eqx.tree_at(
        lambda s: s.observation_sigma,
        states_next_b,
        jnp.broadcast_to(sigma_arr, states_next_b.observation_sigma.shape),
    )
    obs0_b = eqx.tree_at(
        lambda o: o.observation_sigma,
        obs0_b,
        jnp.broadcast_to(sigma_arr, obs0_b.observation_sigma.shape),
    )
    true_ranges = env.true_ranges(states_next_b.positions)  # (B, T, m*k)
    new_ranges = true_ranges + sigma_arr * jax.random.normal(key, true_ranges.shape)
    mask = env.observation_mask(true_ranges.shape[1])
    new_ranges = jnp.where(mask[None, :, None], new_ranges, -1.0)
    sigma_field = jnp.broadcast_to(sigma_arr, true_ranges.shape[:-1])
    obs_next_b = PursuitEvasionObservation(ranges=new_ranges, observation_sigma=sigma_field)
    prev_ranges = jnp.concatenate([obs0_b.ranges[:, None, :], new_ranges[:, :-1, :]], axis=1)
    obs_prev_b = PursuitEvasionObservation(ranges=prev_ranges, observation_sigma=sigma_field)
    return state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b
