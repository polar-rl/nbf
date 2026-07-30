import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jumanji.types import restart, termination, transition

from nbf.base_types import Observation, State

L = 10.0
SENSOR_POS = jnp.array([5.0, 5.0])
SPEED = 0.5
SIGMA_DYN = 0.2
SIGMA_OBS_MIN = 0.1
SIGMA_OBS_MAX = 0.5


def normalize_positions(x):
    return (x - 0.5 * L) / (0.5 * L)


def denormalize_positions(z):
    return z * 0.5 * L + 0.5 * L


DEFAULT_OBSTACLES = jnp.array(
    [
        [2.0, 0.0, 3.0, 7.0],
        [7.0, 3.0, 8.0, 10.0],
        [4.0, 4.0, 6.0, 6.0],
    ]
)

DEFAULT_INTERACTION_RADIUS = 1.5
DEFAULT_INTERACTION_STRENGTH = 1.0


def _reflect_out_of_obstacle(pos: jax.Array, obs: jax.Array) -> jax.Array:
    inside = (pos[0] >= obs[0]) & (pos[0] <= obs[2]) & (pos[1] >= obs[1]) & (pos[1] <= obs[3])
    dists = jnp.array(
        [
            pos[0] - obs[0],
            obs[2] - pos[0],
            pos[1] - obs[1],
            obs[3] - pos[1],
        ]
    )
    min_edge = jnp.argmin(dists)
    reflected = jax.lax.switch(
        min_edge,
        [
            lambda: pos.at[0].set(2.0 * obs[0] - pos[0]),  # entered from left
            lambda: pos.at[0].set(2.0 * obs[2] - pos[0]),  # entered from right
            lambda: pos.at[1].set(2.0 * obs[1] - pos[1]),  # entered from bottom
            lambda: pos.at[1].set(2.0 * obs[3] - pos[1]),  # entered from top
        ],
    )
    return jax.lax.cond(inside, lambda: reflected, lambda: pos)


def clip_to_valid(pos: jax.Array, obstacles: jax.Array) -> jax.Array:
    pos = jnp.where(pos < 0.0, -pos, pos)
    pos = jnp.where(pos > L, 2.0 * L - pos, pos)
    pos = jax.lax.fori_loop(
        0,
        obstacles.shape[0],
        lambda i, p: _reflect_out_of_obstacle(p, obstacles[i]),
        pos,
    )
    return jnp.clip(pos, 0.0, L)


_NUM_SUBSTEPS = 4


def transition_single(position, policy, obstacles, key):
    angle_key, noise_key = jax.random.split(key)
    heading = policy(position, angle_key)
    direction = jnp.array([jnp.cos(heading), jnp.sin(heading)])
    displacement = SPEED * direction + SIGMA_DYN * jax.random.normal(noise_key, (2,))
    step = displacement / _NUM_SUBSTEPS
    return jax.lax.fori_loop(
        0, _NUM_SUBSTEPS, lambda _, p: clip_to_valid(p + step, obstacles), position
    )


def propagate_deterministic(position, policy, obstacles):
    heading = policy(position, jax.random.PRNGKey(0))
    direction = jnp.array([jnp.cos(heading), jnp.sin(heading)])
    displacement = SPEED * direction
    step = displacement / _NUM_SUBSTEPS
    return jax.lax.fori_loop(
        0, _NUM_SUBSTEPS, lambda _, p: clip_to_valid(p + step, obstacles), position
    )


def _pairwise_repulsion(positions, radius, strength):
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = jnp.linalg.norm(diffs, axis=-1)
    safe = jnp.maximum(dists, 1e-6)
    k = positions.shape[0]
    eye = jnp.eye(k, dtype=bool)
    active = (dists < radius) & ~eye
    weights = jnp.where(active, strength * (1.0 - dists / radius), 0.0)
    return jnp.sum((weights / safe)[..., None] * diffs, axis=1)


def transition_joint(positions, policies, obstacles, radius, strength, key):
    """Joint transition for k>=1 evaders with pairwise soft-repulsion coupling.

    `policies` is a tuple of length k of single-evader policy modules; each
    evader uses its own policy and is then nudged away from any neighbours
    closer than `radius`.
    """
    k = positions.shape[0]
    angle_key, noise_key = jax.random.split(key)
    angle_keys = jax.random.split(angle_key, k)
    headings = jnp.stack([policies[i](positions[i], angle_keys[i]) for i in range(k)])
    base_dirs = jnp.stack([jnp.cos(headings), jnp.sin(headings)], axis=-1)

    repulsion = _pairwise_repulsion(positions, radius, strength)
    combined = base_dirs + repulsion
    cnorm = jnp.linalg.norm(combined, axis=-1, keepdims=True)
    direction = combined / jnp.maximum(cnorm, 1e-6)

    noise = SIGMA_DYN * jax.random.normal(noise_key, (k, 2))
    displacement = SPEED * direction + noise
    step = displacement / _NUM_SUBSTEPS

    def body(_, p):
        return jax.vmap(lambda pi, si: clip_to_valid(pi + si, obstacles))(p, step)

    return jax.lax.fori_loop(0, _NUM_SUBSTEPS, body, positions)


def _enumerate_permutations(k: int) -> jax.Array:
    return jnp.asarray(np.array(list(itertools.permutations(range(k))), dtype=np.int32))


class PursuitEvasionState(State):
    positions: jax.Array
    step_count: int
    terminal: bool
    observation_sigma: jax.Array = eqx.field(default_factory=lambda: jnp.asarray(SIGMA_OBS_MAX))

    @property
    def max_feature_value(self) -> int:
        raise ValueError('Pursuit-evasion environment has continuous states')

    def to_flat_features(self) -> jax.Array:
        return normalize_positions(self.positions).reshape(-1)

    def from_flat_features(self, features: jax.Array):
        k = features.shape[0] // 2
        return PursuitEvasionState(
            positions=denormalize_positions(features.reshape(k, 2)),
            step_count=self.step_count,
            terminal=self.terminal,
            observation_sigma=self.observation_sigma,
        )

    def observation_from_state(self) -> Observation:
        ranges = jnp.linalg.norm(self.positions - SENSOR_POS, axis=-1)
        return PursuitEvasionObservation(ranges=ranges, observation_sigma=self.observation_sigma)


class PursuitEvasionObservation(Observation):
    ranges: jax.Array
    observation_sigma: jax.Array = eqx.field(default_factory=lambda: jnp.asarray(SIGMA_OBS_MAX))


class PursuitEvasionGenerator(eqx.Module):
    num_evaders: int = eqx.field(static=True)
    obstacles: jax.Array

    def __call__(self, key: jax.Array) -> PursuitEvasionState:
        pos_key, sigma_key = jax.random.split(key)
        positions = jax.random.uniform(pos_key, (self.num_evaders, 2), minval=0.0, maxval=L)
        positions = jax.vmap(lambda p: clip_to_valid(p, self.obstacles))(positions)
        sigma = jax.random.uniform(sigma_key, (), minval=SIGMA_OBS_MIN, maxval=SIGMA_OBS_MAX)
        return PursuitEvasionState(
            positions=positions,
            step_count=0,
            terminal=False,
            observation_sigma=sigma,
        )


class PursuitEvasionEnv(eqx.Module):
    num_evaders: int = eqx.field(static=True)
    time_limit: int = eqx.field(static=True)
    shuffle_observations: bool = eqx.field(static=True)
    interaction_radius: float = eqx.field(static=True)
    interaction_strength: float = eqx.field(static=True)
    obstacles: jax.Array
    policy: tuple
    perm_indices: jax.Array
    generator: PursuitEvasionGenerator
    has_state_observation: bool = eqx.field(static=True)
    has_continuous_features: bool = eqx.field(static=True)

    def __init__(
        self,
        num_evaders: int = 1,
        time_limit: int = 20,
        obstacles: jax.Array = DEFAULT_OBSTACLES,
        *,
        policy=None,
        shuffle_observations: bool = False,
        interaction_radius: float = DEFAULT_INTERACTION_RADIUS,
        interaction_strength: float = DEFAULT_INTERACTION_STRENGTH,
    ):
        self.num_evaders = num_evaders
        self.time_limit = time_limit
        self.obstacles = obstacles
        self.shuffle_observations = shuffle_observations
        self.interaction_radius = interaction_radius
        self.interaction_strength = interaction_strength
        if policy is None:
            self.policy = tuple()
        elif isinstance(policy, tuple | list):
            if len(policy) != num_evaders:
                raise ValueError(f'Got {len(policy)} policies but num_evaders={num_evaders}')
            self.policy = tuple(policy)
        else:
            self.policy = (policy,) * num_evaders
        self.perm_indices = _enumerate_permutations(num_evaders)
        self.generator = PursuitEvasionGenerator(num_evaders=num_evaders, obstacles=obstacles)
        self.has_state_observation = False
        self.has_continuous_features = True

    def reset(self, key: jax.Array):
        state = self.generator(key)
        obs = PursuitEvasionObservation(
            ranges=jnp.full(self.num_evaders, -1.0),
            observation_sigma=state.observation_sigma,
        )
        return state, restart(observation=obs)

    def step_with_policy(self, state, policy, key):
        if not isinstance(policy, tuple | list):
            policies = (policy,) * self.num_evaders
        else:
            policies = tuple(policy)

        step_key, obs_key, perm_key = jax.random.split(key, 3)
        next_positions = transition_joint(
            state.positions,
            policies,
            self.obstacles,
            self.interaction_radius,
            self.interaction_strength,
            step_key,
        )

        done = state.step_count + 1 >= self.time_limit
        next_state = PursuitEvasionState(
            positions=next_positions,
            step_count=state.step_count + 1,
            terminal=done,
            observation_sigma=state.observation_sigma,
        )

        true_ranges = jnp.linalg.norm(next_positions - SENSOR_POS, axis=-1)
        noise = state.observation_sigma * jax.random.normal(obs_key, true_ranges.shape)
        ranges = true_ranges + noise
        if self.shuffle_observations and self.num_evaders > 1:
            perm = jax.random.permutation(perm_key, self.num_evaders)
            ranges = ranges[perm]
        observation = PursuitEvasionObservation(
            ranges=ranges, observation_sigma=state.observation_sigma
        )

        timestep = jax.lax.cond(done, termination, transition, 0.0, observation)
        return next_state, timestep

    def step(self, state, action, key):
        return self.step_with_policy(state, self.policy, key)

    def observation_weights(self, particles, observations, gold_observation):
        has_obs = gold_observation.ranges[0] >= 0.0
        sigma = gold_observation.observation_sigma
        pred_ranges = jnp.linalg.norm(
            particles.positions - SENSOR_POS,
            axis=-1,
        )

        if self.shuffle_observations and self.num_evaders > 1:
            permuted = gold_observation.ranges[self.perm_indices]
            diff = pred_ranges[:, None, :] - permuted[None, :, :]
            log_per_perm = -0.5 * jnp.sum((diff / sigma) ** 2, axis=-1)
            log_likes = jax.scipy.special.logsumexp(log_per_perm, axis=1)
        else:
            log_likes = -0.5 * jnp.sum(
                ((pred_ranges - gold_observation.ranges) / sigma) ** 2,
                axis=-1,
            )
        n = particles.positions.shape[0]
        weights = jax.lax.cond(
            has_obs,
            lambda: jax.nn.softmax(log_likes),
            lambda: jnp.ones(n) / n,
        )

        # We never discard an observation, so the uniform-fallback flag is always False here.
        return weights, jnp.array(False)

    def state_array(self, state) -> jax.Array:
        return state.positions

    def replace_state_array(self, state, array: jax.Array):
        return eqx.tree_at(lambda s: s.positions, state, array)

    def normalize_flat(self, array: jax.Array) -> jax.Array:
        return normalize_positions(array).reshape(array.shape[0], -1)

    def observation_log_prob(self, array: jax.Array, observation) -> jax.Array:
        sigma = observation.observation_sigma
        pred_ranges = jnp.linalg.norm(array - SENSOR_POS, axis=-1)  # (N, k)
        if self.shuffle_observations and self.num_evaders > 1:
            permuted = observation.ranges[self.perm_indices]  # (P, k)
            diff = pred_ranges[:, None, :] - permuted[None, :, :]
            log_per_perm = -0.5 * jnp.sum((diff / sigma) ** 2, axis=-1)
            return jax.scipy.special.logsumexp(log_per_perm, axis=1)
        return -0.5 * jnp.sum(((pred_ranges - observation.ranges) / sigma) ** 2, axis=-1)

    def constrain(self, array: jax.Array) -> jax.Array:
        return jax.vmap(jax.vmap(lambda p: clip_to_valid(p, self.obstacles)))(array)

    def observation_present(self, observation) -> jax.Array:
        return observation.ranges[0] >= 0.0


def build_pursuit_evasion_env(
    num_evaders: int = 1,
    time_limit: int = 20,
    obstacles: jax.Array = DEFAULT_OBSTACLES,
    *,
    policy=None,
    shuffle_observations: bool = False,
    interaction_radius: float = DEFAULT_INTERACTION_RADIUS,
    interaction_strength: float = DEFAULT_INTERACTION_STRENGTH,
) -> PursuitEvasionEnv:
    return PursuitEvasionEnv(
        num_evaders,
        time_limit,
        obstacles,
        policy=policy,
        shuffle_observations=shuffle_observations,
        interaction_radius=interaction_radius,
        interaction_strength=interaction_strength,
    )
