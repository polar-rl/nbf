import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jumanji.env import Environment

from nbf.base_types import Observation, State


def systematic_indices(weights: jax.Array, key: jax.Array) -> jax.Array:
    n = weights.shape[0]
    cdf = jnp.cumsum(weights)
    x0 = jax.random.uniform(key, (), minval=0.0, maxval=1.0 / n)
    positions = x0 + jnp.arange(n) / n
    return jnp.searchsorted(cdf, positions, side='left')


class ParticleSet(eqx.Module):
    particles: State
    weights: jax.Array
    features: jax.Array | None = None

    @eqx.filter_jit
    def simulate(
        self, env: Environment, action: int, observation: Observation, key: jax.Array
    ) -> tuple['ParticleSet', jax.Array]:
        particles, timesteps = jax.vmap(env.step, in_axes=(0, None, 0))(
            self.particles, action, jax.random.split(key, self.weights.shape[0])
        )

        obs_weights, discarded = env.observation_weights(
            particles, timesteps.observation, observation
        )
        new_weights = self.weights * obs_weights

        weight_sum = jnp.sum(new_weights)
        weights = jax.lax.cond(
            weight_sum > 0.0,
            lambda: new_weights / weight_sum,
            lambda: obs_weights,
        )

        return ParticleSet(particles, weights), discarded

    @eqx.filter_jit
    def neff(self) -> float:
        denominator = jnp.sum(self.weights**2)
        is_defined = (denominator > 0.0) & jnp.isfinite(denominator)

        return jax.lax.cond(is_defined, lambda: 1.0 / denominator, lambda: 0.0)

    @eqx.filter_jit
    def systematic_resample(self, key: jax.Array) -> 'ParticleSet':
        num_particles = self.weights.shape[0]
        indices = systematic_indices(self.weights, key)

        particles = jax.tree.map(lambda x: x[indices], self.particles)
        weights = jnp.full(num_particles, 1 / num_particles)

        return ParticleSet(particles, weights)


class ParticleFilter(eqx.Module):
    """Implement the standard SIR filter with threshold-based resampling."""

    num_particles: int
    resample_factor: float = 0.5

    def __repr__(self) -> str:
        return f'PF ({self.num_particles})'

    @eqx.filter_jit
    def sample(
        self, particle_set: ParticleSet, state: State, num_samples: int, key: jax.Array
    ) -> State:
        probs = particle_set.weights / jnp.sum(particle_set.weights)
        indices = jax.random.choice(key, self.num_particles, (num_samples,), p=probs)

        return jax.tree.map(lambda x: x[indices], particle_set.particles)

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
        key, simulate_key = jax.random.split(key, 2)
        particle_set, simulation_fallback = particle_set.simulate(
            env, action, next_observation, simulate_key
        )

        key, resample_key = jax.random.split(key, 2)
        particle_set = jax.lax.cond(
            particle_set.neff() < self.num_particles * self.resample_factor,
            lambda: particle_set.systematic_resample(resample_key),
            lambda: particle_set,
        )

        # A SIR filter has no sampling pass, so that flag is always False here.
        return particle_set, particle_set.neff(), jnp.array(False), simulation_fallback

    @eqx.filter_jit
    def reset(
        self, env: Environment, state: State, observation: Observation, key: jax.Array
    ) -> ParticleSet:
        keys = jax.random.split(key, self.num_particles)
        if getattr(env, 'has_prior_sampler', False):
            particles = jax.vmap(lambda k: env.sample_prior(state, k))(keys)
        else:
            particles, _ = jax.vmap(env.reset)(keys)
            features = jax.vmap(lambda p: p.to_flat_features())(particles)
            particles = jax.vmap(lambda f: state.from_flat_features(f))(features)
        observations = jax.vmap(lambda p: p.observation_from_state())(particles)
        weights, _ = env.observation_weights(particles, observations, observation)

        return ParticleSet(particles, weights)


class RegularizedParticleFilter(ParticleFilter):
    """SIR with post-resample KDE jitter (Silverman bandwidth on the pre-resample ensemble).

    Uses the shrinkage kernel of Liu & West (2001): particles are contracted toward the
    weighted mean by ``sqrt(1 - h^2)`` before the jitter is added, so the smoothing
    preserves the ensemble covariance.
    """

    bandwidth_scale: float = 1.0

    def __repr__(self) -> str:
        return f'PF-Reg ({self.num_particles})'

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
        key, simulate_key = jax.random.split(key, 2)
        particle_set, simulation_fallback = particle_set.simulate(
            env, action, next_observation, simulate_key
        )
        key, resample_key = jax.random.split(key, 2)
        key, jitter_key = jax.random.split(key, 2)

        parts = env.state_array(particle_set.particles)
        flat = parts.reshape(parts.shape[0], -1)
        n, d = flat.shape
        ok = jnp.all(jnp.isfinite(flat), axis=-1)
        w = particle_set.weights * ok
        w = w / (jnp.sum(w) + 1e-30)
        safe = jnp.where(ok[:, None], flat, 0.0)
        mean = jnp.sum(w[:, None] * safe, axis=0)
        diff = jnp.where(ok[:, None], safe - mean, 0.0)
        cov = jnp.matmul((w[:, None] * diff).T, diff, precision='highest')
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = jnp.linalg.eigh(cov)
        chol = eigvecs * jnp.sqrt(jnp.clip(eigvals, min=0.0))[None, :]
        h = self.bandwidth_scale * (4.0 / (n * (d + 2.0))) ** (1.0 / (d + 4.0))

        shrink = jnp.sqrt(jnp.maximum(1.0 - h * h, 0.0))

        def _resample_and_jitter():
            resampled = particle_set.systematic_resample(resample_key)
            eps = jax.random.normal(jitter_key, (n, d))
            base = env.state_array(resampled.particles).reshape(n, -1)
            jittered = mean + shrink * (base - mean) + h * eps @ chol.T
            particles = env.replace_state_array(
                resampled.particles, env.constrain(jittered.reshape(parts.shape))
            )
            return ParticleSet(particles, resampled.weights)

        particle_set = jax.lax.cond(
            particle_set.neff() < self.num_particles * self.resample_factor,
            _resample_and_jitter,
            lambda: particle_set,
        )
        return particle_set, particle_set.neff(), jnp.array(False), simulation_fallback
