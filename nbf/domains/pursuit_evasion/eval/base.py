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
    propagate_with_key,
    sensor_ranges,
)
from nbf.filters.gmf import GMF

GMF_HEADING_DRAWS = 8


class PEObservationModel(eqx.Module):
    sensors: jax.Array
    num_sensors: int = eqx.field(static=True)
    num_evaders: int = eqx.field(static=True)
    shuffled: bool = eqx.field(static=True)

    def __init__(self, env: PursuitEvasionEnv):
        self.sensors = env.sensors
        self.num_sensors = env.num_sensors
        self.num_evaders = env.num_evaders
        self.shuffled = bool(env.shuffle_observations and env.num_evaders > 1)

    def predicted(self, sigmas: jax.Array, mu: jax.Array) -> jax.Array:
        m, k = self.num_sensors, self.num_evaders
        P = sigmas.shape[0]
        r = sensor_ranges(sigmas.reshape(P, k, 2), self.sensors)  # (P, m, k)
        if self.shuffled:
            order = jnp.argsort(sensor_ranges(mu.reshape(k, 2), self.sensors), axis=-1)  # (m, k)
            r = jnp.take_along_axis(r, jnp.broadcast_to(order[None], r.shape), axis=-1)
        return r.reshape(P, m * k)

    def vector(self, observation) -> jax.Array:
        m, k = self.num_sensors, self.num_evaders
        ranges = observation.ranges.reshape(m, k)
        if self.shuffled:
            ranges = jnp.sort(ranges, axis=-1)
        return ranges.reshape(m * k)

    def noise_std(self, observation) -> jax.Array:
        return observation.observation_sigma


def make_pe_gmf(
    num_evaders: int | None = None,
    M: int = 1,
    observation_sigma_scale: float = 1.0,
    kl_merge: float = 0.5,
    weight_prune: float = 0.01,
    *,
    env: PursuitEvasionEnv | None = None,
    heading_draws: int = GMF_HEADING_DRAWS,
) -> GMF:
    if env is not None:
        num_evaders = env.num_evaders
    n = 2 * num_evaders
    return GMF(
        num_evaders=num_evaders,
        sigma_dyn=SIGMA_DYN,
        sigma_obs=SIGMA_OBS_MAX,
        sensor_pos=SENSOR_POS,
        propagate_fn=propagate_with_key if env is not None else _propagate_fixed_heading,
        init_mu=jnp.full(n, L / 2),
        init_sigma=((L / 2) ** 2 * jnp.eye(n)),
        M=M,
        jitter_scale=0.1 * L,
        kl_merge=kl_merge,
        weight_prune=weight_prune,
        observation_sigma_scale=observation_sigma_scale,
        obs_model=PEObservationModel(env) if env is not None else None,
        heading_draws=heading_draws if env is not None else 0,
    )


def _propagate_fixed_heading(position, policy, obstacles):
    return propagate_with_key(position, policy, obstacles, jax.random.PRNGKey(0))


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
