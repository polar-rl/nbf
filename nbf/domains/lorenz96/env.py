import equinox as eqx
import jax
import jax.numpy as jnp
from jumanji.types import restart, termination, transition

from nbf.base_types import Observation, State

OBS_SENTINEL = -1e4
_OBS_PRESENT_THRESHOLD = -1e3


def l96_advection(x):
    return (jnp.roll(x, -1, -1) - jnp.roll(x, 2, -1)) * jnp.roll(x, 1, -1)


def l96_tendency(x, forcing):
    return l96_advection(x) - x + forcing


def linear_tendency(x, forcing):
    return -x + forcing


def rk4_window(x, forcing, dt, substeps, linear=False):
    f = linear_tendency if linear else l96_tendency

    def body(_, x):
        k1 = f(x, forcing)
        k2 = f(x + 0.5 * dt * k1, forcing)
        k3 = f(x + 0.5 * dt * k2, forcing)
        k4 = f(x + dt * k3, forcing)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return jax.lax.fori_loop(0, substeps, body, x)


def conditional_gaussian_redraw(means, obs_values, obs_idx, sigma_dyn, sigma_obs, key):
    if sigma_dyn <= 0:
        raise ValueError('conditional_gaussian_redraw requires sigma_dyn > 0')
    noise = jax.random.normal(key, means.shape)
    x = means + sigma_dyn * noise
    post_var = 1.0 / (1.0 / sigma_dyn**2 + 1.0 / sigma_obs**2)
    post_mean = post_var * (means[..., obs_idx] / sigma_dyn**2 + obs_values / sigma_obs**2)
    x_obs = post_mean + jnp.sqrt(post_var) * noise[..., obs_idx]
    return x.at[..., obs_idx].set(x_obs)


def rw_mh_redraw(x, means, obs_values, obs_idx, sigma_dyn, sigma_obs, num_steps, step_size, key):
    if sigma_dyn <= 0:
        raise ValueError('rw_mh_redraw requires sigma_dyn > 0')

    def log_target(x):
        lp_trans = -0.5 * jnp.sum(((x - means) / sigma_dyn) ** 2, axis=-1)
        lp_obs = -0.5 * jnp.sum(((x[..., obs_idx] - obs_values) / sigma_obs) ** 2, axis=-1)
        return lp_trans + lp_obs

    def body(carry, key_iter):
        x, cur_log = carry
        prop_key, acc_key = jax.random.split(key_iter)
        prop = x + step_size * jax.random.normal(prop_key, x.shape)
        prop_log = log_target(prop)
        u = jax.random.uniform(acc_key, prop_log.shape)
        accept = jnp.log(u + 1e-30) < prop_log - cur_log
        new_x = jnp.where(accept[..., None], prop, x)
        new_log = jnp.where(accept, prop_log, cur_log)
        return (new_x, new_log), None

    (x, _), _ = jax.lax.scan(body, (x, log_target(x)), jax.random.split(key, num_steps))
    return x


class Lorenz96State(State):
    x: jax.Array
    step_count: int
    terminal: bool
    forcing: jax.Array
    sigma_obs: jax.Array
    center: jax.Array
    norm_mean: jax.Array
    norm_std: jax.Array
    obs_idx: jax.Array

    @property
    def max_feature_value(self) -> int:
        raise ValueError('Lorenz-96 environment has continuous states')

    def to_flat_features(self) -> jax.Array:
        return (self.x - self.norm_mean) / self.norm_std

    def from_flat_features(self, features: jax.Array):
        return eqx.tree_at(lambda s: s.x, self, features * self.norm_std + self.norm_mean)

    def observation_from_state(self) -> Observation:
        return Lorenz96Observation(values=self.x[..., self.obs_idx], sigma_obs=self.sigma_obs)


class Lorenz96Observation(Observation):
    values: jax.Array
    sigma_obs: jax.Array


class Lorenz96Generator(eqx.Module):
    K: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    substeps: int = eqx.field(static=True)
    spinup_windows: int = eqx.field(static=True)
    sigma_obs_min: float = eqx.field(static=True)
    sigma_obs_max: float = eqx.field(static=True)
    linear: bool = eqx.field(static=True)
    f_bands: jax.Array
    norm_mean: jax.Array
    norm_std: jax.Array
    obs_idx: jax.Array

    def __call__(self, key: jax.Array) -> Lorenz96State:
        band_key, f_key, sigma_key, init_key, delta_key = jax.random.split(key, 5)
        band = self.f_bands[jax.random.randint(band_key, (), 0, self.f_bands.shape[0])]
        forcing = jax.random.uniform(f_key, (), minval=band[0], maxval=band[1])
        sigma_obs = jax.random.uniform(
            sigma_key, (), minval=self.sigma_obs_min, maxval=self.sigma_obs_max
        )
        x0 = forcing + 0.5 * jax.random.normal(init_key, (self.K,))
        x = jax.lax.fori_loop(
            0,
            self.spinup_windows,
            lambda _, xi: rk4_window(xi, forcing, self.dt, self.substeps, self.linear),
            x0,
        )
        center = x + jax.random.normal(delta_key, (self.K,))
        return Lorenz96State(
            x=x,
            step_count=0,
            terminal=False,
            forcing=forcing,
            sigma_obs=sigma_obs,
            center=center,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            obs_idx=self.obs_idx,
        )


class Lorenz96Env(eqx.Module):
    K: int = eqx.field(static=True)
    time_limit: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    substeps: int = eqx.field(static=True)
    sigma_dyn: float = eqx.field(static=True)
    sigma_init: float = eqx.field(static=True)
    linear: bool = eqx.field(static=True)
    obs_sites: tuple = eqx.field(static=True)
    norm_mean: jax.Array
    norm_std: jax.Array
    generator: Lorenz96Generator
    has_state_observation: bool = eqx.field(static=True)
    has_continuous_features: bool = eqx.field(static=True)
    has_prior_sampler: bool = eqx.field(static=True)

    def __init__(
        self,
        K: int = 20,
        time_limit: int = 20,
        *,
        dt: float = 0.01,
        substeps: int = 5,
        sigma_dyn: float = 0.25,
        sigma_obs_min: float = 0.5,
        sigma_obs_max: float = 1.5,
        f_bands=((5.0, 7.0), (9.0, 11.0)),
        spinup_time: float = 10.0,
        sigma_init: float = 2.0,
        norm_mean: float = 0.0,
        norm_std: float = 1.0,
        linear: bool = False,
        obs_sites: tuple | None = None,
    ):
        if obs_sites is None:
            if K % 2 != 0:
                raise ValueError(f'K must be even (odd-site observations), got {K}')
            obs_sites = tuple(range(1, K, 2))
        if any(i < 0 or i >= K for i in obs_sites):
            raise ValueError(f'obs_sites must lie in [0, {K}), got {obs_sites}')
        self.obs_sites = tuple(obs_sites)
        self.K = K
        self.time_limit = time_limit
        self.dt = dt
        self.substeps = substeps
        self.sigma_dyn = sigma_dyn
        self.sigma_init = sigma_init
        self.linear = linear
        self.norm_mean = jnp.asarray(norm_mean)
        self.norm_std = jnp.asarray(norm_std)
        self.generator = Lorenz96Generator(
            K=K,
            dt=dt,
            substeps=substeps,
            spinup_windows=int(round(spinup_time / (dt * substeps))),
            sigma_obs_min=sigma_obs_min,
            sigma_obs_max=sigma_obs_max,
            linear=linear,
            f_bands=jnp.asarray(f_bands),
            norm_mean=jnp.asarray(norm_mean),
            norm_std=jnp.asarray(norm_std),
            obs_idx=jnp.asarray(self.obs_sites),
        )
        self.has_state_observation = False
        self.has_continuous_features = True
        self.has_prior_sampler = True

    @property
    def obs_idx(self) -> jax.Array:
        return jnp.asarray(self.obs_sites)

    @property
    def num_obs(self) -> int:
        return len(self.obs_sites)

    def reset(self, key: jax.Array):
        state = self.generator(key)
        obs = Lorenz96Observation(
            values=jnp.full(self.num_obs, OBS_SENTINEL), sigma_obs=state.sigma_obs
        )
        return state, restart(observation=obs)

    def step(self, state, action, key):
        step_key, obs_key = jax.random.split(key)
        x = rk4_window(state.x, state.forcing, self.dt, self.substeps, self.linear)
        x = x + self.sigma_dyn * jax.random.normal(step_key, x.shape)
        done = state.step_count + 1 >= self.time_limit
        next_state = eqx.tree_at(
            lambda s: (s.x, s.step_count, s.terminal),
            state,
            (x, state.step_count + 1, done),
        )
        values = x[self.obs_idx] + state.sigma_obs * jax.random.normal(obs_key, (self.num_obs,))
        observation = Lorenz96Observation(values=values, sigma_obs=state.sigma_obs)
        timestep = jax.lax.cond(done, termination, transition, 0.0, observation)
        return next_state, timestep

    def sample_prior(self, state, key):
        x = state.center + self.sigma_init * jax.random.normal(key, state.center.shape)
        return eqx.tree_at(lambda s: s.x, state, x)

    def observation_weights(self, particles, observations, gold_observation):
        has_obs = gold_observation.values[0] > _OBS_PRESENT_THRESHOLD
        log_likes = -0.5 * jnp.sum(
            ((particles.x[:, self.obs_idx] - gold_observation.values) / gold_observation.sigma_obs)
            ** 2,
            axis=-1,
        )
        finite = jnp.isfinite(log_likes)
        log_likes = jnp.where(finite, log_likes, -jnp.inf)
        n = particles.x.shape[0]
        weights = jax.lax.cond(
            has_obs & jnp.any(finite),
            lambda: jax.nn.softmax(log_likes),
            lambda: jnp.ones(n) / n,
        )
        return weights, jnp.array(False)

    def state_array(self, state) -> jax.Array:
        return state.x

    def replace_state_array(self, state, array: jax.Array):
        return eqx.tree_at(lambda s: s.x, state, array)

    def normalize_flat(self, array: jax.Array) -> jax.Array:
        return ((array - self.norm_mean) / self.norm_std).reshape(array.shape[0], -1)

    def observation_log_prob(self, array: jax.Array, observation) -> jax.Array:
        return -0.5 * jnp.sum(
            ((array[:, self.obs_idx] - observation.values) / observation.sigma_obs) ** 2,
            axis=-1,
        )

    def propagate_mean_array(self, state) -> jax.Array:
        return rk4_window(state.x, state.forcing, self.dt, self.substeps, self.linear)

    def conjugate_move(self, means: jax.Array, observation, key: jax.Array) -> jax.Array:
        return conditional_gaussian_redraw(
            means, observation.values, self.obs_idx, self.sigma_dyn, observation.sigma_obs, key
        )

    def transition_log_prob(self, propagated_means: jax.Array, array: jax.Array) -> jax.Array:
        if self.sigma_dyn <= 0:
            raise ValueError('transition_log_prob requires sigma_dyn > 0')
        return -0.5 * jnp.sum(((array - propagated_means) / self.sigma_dyn) ** 2, axis=-1)

    def constrain(self, array: jax.Array) -> jax.Array:
        return array

    def observation_present(self, observation) -> jax.Array:
        return observation.values[0] > _OBS_PRESENT_THRESHOLD


def build_lorenz96_env(data_cfg, *, time_limit=None, linear=False, obs_sites=None) -> Lorenz96Env:
    return Lorenz96Env(
        K=data_cfg.K,
        time_limit=data_cfg.episode_length if time_limit is None else time_limit,
        dt=data_cfg.dt,
        substeps=data_cfg.window_substeps,
        sigma_dyn=data_cfg.sigma_dyn,
        sigma_obs_min=data_cfg.sigma_obs_min,
        sigma_obs_max=data_cfg.sigma_obs_max,
        f_bands=tuple(tuple(b) for b in data_cfg.f_bands),
        spinup_time=data_cfg.spinup_time,
        sigma_init=data_cfg.sigma_init,
        norm_mean=data_cfg.norm_mean if data_cfg.norm_mean is not None else 0.0,
        norm_std=data_cfg.norm_std if data_cfg.norm_std is not None else 1.0,
        linear=linear,
        obs_sites=obs_sites,
    )
