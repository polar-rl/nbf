import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jumanji.types import restart, termination, transition

from nbf.base_types import Observation, State

L = 10.0
SENSOR_POS = jnp.array([5.0, 5.0])
SENSOR_LAYOUT = jnp.array([[5.0, 5.0], [1.0, 1.0], [9.0, 9.0], [1.0, 9.0], [9.0, 1.0]])
FEATURE_MODES = (
    'nats',
    'range',
    'both',
)  # units of the likelihood feature; see likelihood_features


def sensor_positions(num_sensors: int) -> jax.Array:
    if not 1 <= num_sensors <= SENSOR_LAYOUT.shape[0]:
        raise ValueError(f'num_sensors must be in 1..{SENSOR_LAYOUT.shape[0]}, got {num_sensors}')
    return SENSOR_LAYOUT[:num_sensors]


def sensor_ranges(positions: jax.Array, sensors: jax.Array) -> jax.Array:
    """(..., k, 2) positions, (m, 2) sensors -> (..., m, k) ranges."""
    return jnp.linalg.norm(positions[..., None, :, :] - sensors[:, None, :], axis=-1)


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


def propagate_with_key(position, policy, obstacles, key):
    heading = policy(position, key)
    direction = jnp.array([jnp.cos(heading), jnp.sin(heading)])
    displacement = SPEED * direction
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
    sigma_obs_min: float = eqx.field(static=True, default=SIGMA_OBS_MIN)
    sigma_obs_max: float = eqx.field(static=True, default=SIGMA_OBS_MAX)

    def __call__(self, key: jax.Array) -> PursuitEvasionState:
        pos_key, sigma_key = jax.random.split(key)
        positions = jax.random.uniform(pos_key, (self.num_evaders, 2), minval=0.0, maxval=L)
        positions = jax.vmap(lambda p: clip_to_valid(p, self.obstacles))(positions)
        sigma = jax.random.uniform(
            sigma_key, (), minval=self.sigma_obs_min, maxval=self.sigma_obs_max
        )
        return PursuitEvasionState(
            positions=positions,
            step_count=0,
            terminal=False,
            observation_sigma=sigma,
        )


class PursuitEvasionEnv(eqx.Module):
    num_evaders: int = eqx.field(static=True)
    num_sensors: int = eqx.field(static=True)
    obs_interval: int = eqx.field(static=True)
    feature_mode: str = eqx.field(static=True)
    time_limit: int = eqx.field(static=True)
    shuffle_observations: bool = eqx.field(static=True)
    interaction_radius: float = eqx.field(static=True)
    interaction_strength: float = eqx.field(static=True)
    obstacles: jax.Array
    sensors: jax.Array
    policy: tuple
    perm_indices: jax.Array
    generator: PursuitEvasionGenerator
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
        num_sensors: int = 1,
        obs_interval: int = 1,
        sigma_obs_min: float = SIGMA_OBS_MIN,
        sigma_obs_max: float = SIGMA_OBS_MAX,
        feature_mode: str = 'nats',
    ):
        if int(obs_interval) < 1:
            raise ValueError(f'obs_interval must be >= 1, got {obs_interval}')
        if feature_mode not in FEATURE_MODES:
            raise ValueError(f'feature_mode must be one of {FEATURE_MODES}, got {feature_mode!r}')
        self.feature_mode = feature_mode
        if not 0.0 < sigma_obs_min <= sigma_obs_max:
            raise ValueError(
                f'need 0 < sigma_obs_min <= sigma_obs_max, got {sigma_obs_min}, {sigma_obs_max}'
            )
        self.num_evaders = num_evaders
        self.num_sensors = int(num_sensors)
        self.obs_interval = int(obs_interval)
        self.sensors = sensor_positions(self.num_sensors)
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
        self.generator = PursuitEvasionGenerator(
            num_evaders=num_evaders,
            obstacles=obstacles,
            sigma_obs_min=float(sigma_obs_min),
            sigma_obs_max=float(sigma_obs_max),
        )
        self.has_continuous_features = True

    def reset(self, key: jax.Array):
        state = self.generator(key)
        obs = PursuitEvasionObservation(
            ranges=jnp.full(self.num_sensors * self.num_evaders, -1.0),
            observation_sigma=state.observation_sigma,
        )
        return state, restart(observation=obs)

    @property
    def _shuffled(self) -> bool:
        return self.shuffle_observations and self.num_evaders > 1

    def observed_step(self, step_count) -> jax.Array:
        return (jnp.asarray(step_count) % self.obs_interval) == 0

    def observation_mask(self, T: int) -> jax.Array:
        """(T,) bool: which of the steps 1..T carry an observation."""
        return self.observed_step(jnp.arange(1, T + 1))

    def true_ranges(self, positions: jax.Array) -> jax.Array:
        """(..., k, 2) positions -> (..., m*k) noise-free ranges in sensor-major order."""
        r = sensor_ranges(positions, self.sensors)
        return r.reshape(*r.shape[:-2], self.num_sensors * self.num_evaders)

    def shuffle_ranges(self, ranges: jax.Array, key: jax.Array) -> jax.Array:
        """Permute each sensor's k ranges independently; identity when not shuffling."""
        if not self._shuffled:
            return ranges
        m, k = self.num_sensors, self.num_evaders
        perms = jax.vmap(lambda kk: jax.random.permutation(kk, k))(jax.random.split(key, m))
        return jnp.take_along_axis(ranges.reshape(m, k), perms, axis=1).reshape(m * k)

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

        true_ranges = self.true_ranges(next_positions)
        noise = state.observation_sigma * jax.random.normal(obs_key, true_ranges.shape)
        ranges = self.shuffle_ranges(true_ranges + noise, perm_key)
        ranges = jnp.where(self.observed_step(state.step_count + 1), ranges, -1.0)
        observation = PursuitEvasionObservation(
            ranges=ranges, observation_sigma=state.observation_sigma
        )

        timestep = jax.lax.cond(done, termination, transition, 0.0, observation)
        return next_state, timestep

    def step(self, state, action, key):
        return self.step_with_policy(state, self.policy, key)

    def observation_weights(self, particles, observations, gold_observation):
        has_obs = gold_observation.ranges[0] >= 0.0
        log_likes = self.observation_log_prob(particles.positions, gold_observation)
        n = particles.positions.shape[0]
        weights = jax.lax.cond(
            has_obs,
            lambda: jax.nn.softmax(log_likes),
            lambda: jnp.ones(n) / n,
        )

        # We never discard an observation, so the uniform-fallback flag is always False here.
        return weights, jnp.array(False)

    def state_observation_weights(self, particles, observation) -> tuple[jax.Array, jax.Array]:
        n = particles.positions.shape[0]
        return jnp.full(n, 1.0 / n), jnp.array(False)

    @property
    def likelihood_feature_dim(self) -> int:
        base = 1 if self._shuffled else self.num_sensors * self.num_evaders
        return 2 * base if self.feature_mode == 'both' else base

    @property
    def likelihood_feature_index(self) -> jax.Array:
        """Position variable (evader e, coordinate) -> that evader's first sensor column;
        one shared column when the ranges are shuffled."""
        evader = jnp.arange(2 * self.num_evaders) // 2
        return jnp.zeros_like(evader) if self._shuffled else evader

    def likelihood_features(self, particles, observation, weights=None) -> jax.Array:
        """Per-particle likelihood features, zero when no observation is present.

        Labelled ranges factor per (sensor, evader): one residual each. Shuffled ranges couple
        all evaders through every sensor's assignment sum: one scalar, the log-likelihood
        shifted so the best particle sits at 0.

        `feature_mode` picks the units. 'nats': standardized residuals clipped at
        RESIDUAL_CLIP sigma (labelled) or the shifted log-likelihood floored at
        LOG_FEATURE_FLOOR nats (shuffled). 'range': the same
        quantities in range units, residual r_obs - r_pred (labelled) or minus the root of
        the excess squared range residual over the best particle (shuffled), clipped at
        RESIDUAL_CLIP range units, so at small sigma the encoder still sees the graded
        landscape instead of one particle above a floor. 'both' concatenates the two."""
        from nbf.filters.likelihood import LOG_FEATURE_FLOOR, RESIDUAL_CLIP

        has_obs = self.observation_present(observation)
        sigma = observation.observation_sigma
        blocks = []
        if self._shuffled:
            lp = self.observation_log_prob(particles.positions, observation)
            excess = lp - jnp.max(lp)  # <= 0, nats
            if self.feature_mode in ('nats', 'both'):
                blocks.append(jnp.maximum(excess, LOG_FEATURE_FLOOR)[:, None])
            if self.feature_mode in ('range', 'both'):
                rng = -jnp.sqrt(jnp.maximum(-2.0 * sigma**2 * excess, 0.0))
                blocks.append(jnp.maximum(rng, -RESIDUAL_CLIP)[:, None])
        else:
            resid = observation.ranges - self.true_ranges(particles.positions)  # (n, m*k)
            if self.feature_mode in ('nats', 'both'):
                blocks.append(jnp.clip(resid / sigma, -RESIDUAL_CLIP, RESIDUAL_CLIP))
            if self.feature_mode in ('range', 'both'):
                blocks.append(jnp.clip(resid, -RESIDUAL_CLIP, RESIDUAL_CLIP))
        feat = jnp.concatenate(blocks, axis=-1)
        return jnp.where(has_obs, feat, 0.0)

    def state_array(self, state) -> jax.Array:
        return state.positions

    def replace_state_array(self, state, array: jax.Array):
        return eqx.tree_at(lambda s: s.positions, state, array)

    def normalize_flat(self, array: jax.Array) -> jax.Array:
        return normalize_positions(array).reshape(array.shape[0], -1)

    def normalize_log_jacobian(self) -> jax.Array:
        """log |dz/dx| of `normalize_flat` (z = (x - L/2) / (L/2) per coordinate)."""
        return -2.0 * self.num_evaders * jnp.log(0.5 * L)

    def observation_log_prob(self, array: jax.Array, observation) -> jax.Array:
        """(N, k, 2) positions -> (N,) log-likelihood up to a constant. With shuffled
        ranges each sensor contributes a log-sum over the k! assignments of its ranges to
        the evaders; the sensors' terms add, and every term depends on all k positions."""
        sigma = observation.observation_sigma
        m, k = self.num_sensors, self.num_evaders
        pred_ranges = sensor_ranges(array, self.sensors)  # (N, m, k)
        if self._shuffled:
            obs = observation.ranges.reshape(m, k)
            permuted = obs[:, self.perm_indices]  # (m, P, k)
            diff = pred_ranges[:, :, None, :] - permuted[None, :, :, :]  # (N, m, P, k)
            log_per_perm = -0.5 * jnp.sum((diff / sigma) ** 2, axis=-1)
            return jnp.sum(jax.scipy.special.logsumexp(log_per_perm, axis=2), axis=1)
        obs = observation.ranges.reshape(m, k)
        return -0.5 * jnp.sum(((pred_ranges - obs) / sigma) ** 2, axis=(1, 2))

    def constrain(self, array: jax.Array) -> jax.Array:
        return jax.vmap(jax.vmap(lambda p: clip_to_valid(p, self.obstacles)))(array)

    def observation_present(self, observation) -> jax.Array:
        return observation.ranges[0] >= 0.0

    def _canonical_ranges(self, ranges: jax.Array) -> jax.Array:
        """(..., m*k) ranges sorted within each sensor's block when shuffled."""
        if not self._shuffled:
            return ranges
        m, k = self.num_sensors, self.num_evaders
        blocks = ranges.reshape(*ranges.shape[:-1], m, k)
        return jnp.sort(blocks, axis=-1).reshape(*ranges.shape[:-1], m * k)

    def predicted_observation(self, flat: jax.Array) -> jax.Array:
        """(N, 2k) flat positions -> (N, m*k) ranges in the order the operator compares."""
        pos = flat.reshape(flat.shape[0], self.num_evaders, 2)
        return self._canonical_ranges(self.true_ranges(pos))

    def observation_vector(self, observation) -> jax.Array:
        return self._canonical_ranges(observation.ranges)

    def observation_noise_std(self, observation) -> jax.Array:
        return observation.observation_sigma


def build_pursuit_evasion_env(
    num_evaders: int = 1,
    time_limit: int = 20,
    obstacles: jax.Array = DEFAULT_OBSTACLES,
    *,
    policy=None,
    shuffle_observations: bool = False,
    interaction_radius: float = DEFAULT_INTERACTION_RADIUS,
    interaction_strength: float = DEFAULT_INTERACTION_STRENGTH,
    num_sensors: int = 1,
    obs_interval: int = 1,
    sigma_obs_min: float = SIGMA_OBS_MIN,
    sigma_obs_max: float = SIGMA_OBS_MAX,
    feature_mode: str = 'nats',
) -> PursuitEvasionEnv:
    return PursuitEvasionEnv(
        num_evaders,
        time_limit,
        obstacles,
        policy=policy,
        shuffle_observations=shuffle_observations,
        interaction_radius=interaction_radius,
        interaction_strength=interaction_strength,
        num_sensors=num_sensors,
        obs_interval=obs_interval,
        sigma_obs_min=sigma_obs_min,
        sigma_obs_max=sigma_obs_max,
        feature_mode=feature_mode,
    )
