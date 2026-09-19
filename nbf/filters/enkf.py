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


OBSERVATION_OPERATOR_HOOKS = (
    'predicted_observation',
    'observation_vector',
    'observation_noise_std',
)


class EnsembleKalmanFilter(eqx.Module):
    """Stochastic (perturbed-observation) EnKF, Burgers et al. (1998).

    Two analysis paths, depending on the environment:
    - `env.obs_idx` present (Lorenz-96): the observation reads coordinates directly, and
      the ensemble covariance is tapered by Gaspari-Cohn at `loc_radius` on the ring.
    - otherwise the environment supplies an observation operator through the hooks in
      `OBSERVATION_OPERATOR_HOOKS` (pursuit-evasion: sorted ranges). The analysis uses
      the ensemble cross-covariance between states and predicted observations, with no
      localization, and the analysed members pass through `env.constrain` when present.
    """

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

    def _analysis_operator(self, env: Environment, X: jax.Array, observation, key: jax.Array):
        m = X.shape[0]
        xbar = jnp.mean(X, axis=0)
        A = (X - xbar) * self.inflation
        X = xbar + A
        Z = env.predicted_observation(X)
        y = env.observation_vector(observation)
        sigma = env.observation_noise_std(observation)
        Yb = Z - jnp.mean(Z, axis=0)
        Pxz = A.T @ Yb / (m - 1)
        S = Yb.T @ Yb / (m - 1) + sigma**2 * jnp.eye(Z.shape[1])
        eta = sigma * jax.random.normal(key, Z.shape)
        eta = eta - jnp.mean(eta, axis=0)
        innovations = y + eta - Z
        return X + innovations @ jnp.linalg.solve(S, Pxz.T)

    def _analyse(self, env: Environment, X: jax.Array, observation, key: jax.Array):
        if hasattr(env, 'obs_idx'):
            return self._analysis(X, observation, env.obs_idx, key)
        missing = [h for h in OBSERVATION_OPERATOR_HOOKS if not hasattr(env, h)]
        if missing:
            raise ValueError(f'{self!r} requires env.obs_idx or the hooks {missing}')
        shape = X.shape
        flat = self._analysis_operator(env, X.reshape(shape[0], -1), observation, key)
        X = flat.reshape(shape)
        return env.constrain(X) if hasattr(env, 'constrain') else X

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
            lambda: self._analyse(env, X, next_observation, perturb_key),
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


class LocalEnsembleTransformKalmanFilter(EnsembleKalmanFilter):
    """LETKF (Hunt et al. 2007): deterministic ensemble transform with
    observation-space localization, solved independently per grid point."""

    def __repr__(self) -> str:
        return f'LETKF ({self.num_members})'

    def _analysis_operator(self, env, X, observation, key):
        raise ValueError('LETKF localizes by site and needs env.obs_idx')

    def _analysis(self, X: jax.Array, observation: Observation, obs_idx, key: jax.Array):
        del key
        m, K = X.shape
        xbar = jnp.mean(X, axis=0)
        A = (X - xbar) * self.inflation
        Yb = A[:, obs_idx]
        d = observation.values - xbar[obs_idx]
        rho = ring_taper(K, obs_idx, self.loc_radius)
        inv_r = 1.0 / observation.sigma_obs**2
        eye = (m - 1) * jnp.eye(m)

        def _local(rho_g, a_g):
            C = Yb * (rho_g * inv_r)[None, :]
            evals, evecs = jnp.linalg.eigh(eye + C @ Yb.T)
            evals = jnp.maximum(evals, 1e-6)
            P = (evecs / evals) @ evecs.T
            W = (evecs * jnp.sqrt((m - 1) / evals)) @ evecs.T
            wbar = P @ (C @ d)
            return (W + wbar[None, :]) @ a_g

        return xbar + jax.vmap(_local, in_axes=(0, 1), out_axes=1)(rho, A)
