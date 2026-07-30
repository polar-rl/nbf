import equinox as eqx
import jax
import jax.numpy as jnp
from jumanji.env import Environment

from nbf.base_types import Observation, State
from nbf.filters.particle import ParticleSet


def gaspari_cohn(r):
    r = jnp.abs(r)
    near = -0.25 * r**5 + 0.5 * r**4 + 0.625 * r**3 - (5.0 / 3.0) * r**2 + 1.0
    far = (
        (1.0 / 12.0) * r**5
        - 0.5 * r**4
        + 0.625 * r**3
        + (5.0 / 3.0) * r**2
        - 5.0 * r
        + 4.0
        - 2.0 / (3.0 * jnp.maximum(r, 1e-10))
    )
    return jnp.where(r <= 1.0, near, jnp.where(r <= 2.0, far, 0.0))


def ring_taper(K: int, obs_idx: jax.Array, radius: float) -> jax.Array:
    d = jnp.abs(jnp.arange(K)[:, None] - obs_idx[None, :])
    d = jnp.minimum(d, K - d)
    return gaspari_cohn(d / radius)


class EnsembleKalmanFilter(eqx.Module):
    num_members: int
    inflation: float = 1.05
    loc_radius: float = 4.0

    def __repr__(self) -> str:
        return f'EnKF ({self.num_members})'

    @eqx.filter_jit
    def sample(
        self, particle_set: ParticleSet, state: State, num_samples: int, key: jax.Array
    ) -> State:
        indices = jax.random.randint(key, (num_samples,), 0, self.num_members)
        return jax.tree.map(lambda x: x[indices], particle_set.particles)

    def _analysis(self, X: jax.Array, observation: Observation, obs_idx, key: jax.Array):
        m, K = X.shape
        xbar = jnp.mean(X, axis=0)
        A = (X - xbar) * self.inflation
        X = xbar + A
        C = A.T @ A / (m - 1)
        taper = ring_taper(K, obs_idx, self.loc_radius)
        PH = taper * C[:, obs_idx]
        S = taper[obs_idx, :] * C[obs_idx][:, obs_idx] + observation.sigma_obs**2 * jnp.eye(
            obs_idx.shape[0]
        )
        eta = observation.sigma_obs * jax.random.normal(key, (m, obs_idx.shape[0]))
        eta = eta - jnp.mean(eta, axis=0)
        innovations = observation.values + eta - X[:, obs_idx]
        return X + innovations @ jnp.linalg.solve(S, PH.T)

    @eqx.filter_jit
    def update(
        self,
        env: Environment,
        particle_set: ParticleSet,
        state: State,
        observation: Observation,
        action: int,
        next_observation: Observation,
        key: jax.Array,
    ) -> tuple[ParticleSet, jax.Array, jax.Array, jax.Array]:
        forecast_key, perturb_key = jax.random.split(key)
        particles, _ = jax.vmap(env.step, in_axes=(0, None, 0))(
            particle_set.particles, action, jax.random.split(forecast_key, self.num_members)
        )
        X = env.state_array(particles)
        X = jax.lax.cond(
            env.observation_present(next_observation),
            lambda: self._analysis(X, next_observation, env.obs_idx, perturb_key),
            lambda: X,
        )
        particles = env.replace_state_array(particles, X)
        weights = jnp.full(self.num_members, 1.0 / self.num_members)
        new_set = ParticleSet(particles, weights)
        return new_set, new_set.neff(), jnp.array(False), jnp.array(False)

    @eqx.filter_jit
    def reset(
        self, env: Environment, state: State, observation: Observation, key: jax.Array
    ) -> ParticleSet:
        keys = jax.random.split(key, self.num_members)
        if getattr(env, 'has_prior_sampler', False):
            particles = jax.vmap(lambda k: env.sample_prior(state, k))(keys)
        else:
            particles, _ = jax.vmap(env.reset)(keys)
            features = jax.vmap(lambda p: p.to_flat_features())(particles)
            particles = jax.vmap(lambda f: state.from_flat_features(f))(features)
        weights = jnp.full(self.num_members, 1.0 / self.num_members)
        return ParticleSet(particles, weights)
