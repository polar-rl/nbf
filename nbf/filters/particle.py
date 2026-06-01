import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jumanji.env import Environment

from nbf.base_types import Observation, State


class ParticleSet(eqx.Module):
    particles: State
    weights: jax.Array

    @eqx.filter_jit
    def simulate(
        self, env: Environment, action: int, observation: Observation, key: jax.Array
    ) -> 'ParticleSet':
        sim_keys = jax.random.split(key, self.weights.shape[0])
        particles, timesteps = jax.vmap(env.step, in_axes=(0, None, 0))(
            self.particles, action, sim_keys
        )
        obs_weights = env.observation_weights(particles, timesteps.observation, observation)
        new_weights = self.weights * obs_weights
        weight_sum = jnp.sum(new_weights)
        weights = jax.lax.cond(
            weight_sum > 0.0,
            lambda: new_weights / weight_sum,
            lambda: obs_weights,
        )
        return ParticleSet(particles, weights)

    @eqx.filter_jit
    def neff(self) -> float:
        denominator = jnp.sum(self.weights**2)
        is_defined = (denominator > 0.0) & jnp.isfinite(denominator)

        return jax.lax.cond(is_defined, lambda: 1.0 / denominator, lambda: 0.0)

    @eqx.filter_jit
    def systematic_resample(self, key: jax.Array) -> 'ParticleSet':
        num_particles = self.weights.shape[0]

        cdf = jnp.cumsum(self.weights)
        x0 = jax.random.uniform(key, (), minval=0.0, maxval=1.0 / num_particles)
        positions = x0 + jnp.arange(num_particles) / num_particles
        indices = jnp.searchsorted(cdf, positions, side='left')

        particles = jax.tree.map(lambda x: x[indices], self.particles)
        weights = jnp.full(num_particles, 1 / num_particles)

        return ParticleSet(particles, weights)


# ---------------------------------------------------------------------------
# PF KDE bandwidth lookup
# ---------------------------------------------------------------------------
#
# Recommended PF KDE bandwidths from prior sweeps. Keyed by ``(k, num_particles)``.
# Maintain by hand. Falls back
# to nearest particle count for the same ``k`` (log-distance), then to
# ``DEFAULT_PF_KDE_BANDWIDTH``.

DEFAULT_PF_KDE_BANDWIDTH = 0.3

RECOMMENDED_PF_KDE_BANDWIDTH: dict[tuple[int, int], float] = {
    (1, 64): 0.50,
    (1, 128): 0.30,
    (1, 256): 0.30,
    (1, 512): 0.50,
    (1, 2048): 0.20,
    (1, 4096): 0.10,
    (1, 8192): 0.10,
    (2, 128): 1.00,
    (2, 512): 1.50,
    (2, 1024): 0.50,
    (2, 2048): 1.00,
    (2, 4096): 0.75,
    (2, 8192): 0.50,
    (3, 512): 1.50,
    (3, 1024): 0.75,
    (3, 2048): 1.50,
    (3, 4096): 1.00,
    (3, 8192): 1.00,
    (5, 512): 2.00,
    (5, 2048): 2.00,
    (5, 4096): 2.00,
    (5, 8192): 2.00,
}


def recommended_pf_bandwidth(k: int, num_particles: int) -> float:
    """Return the swept-optimal PF KDE bandwidth for ``(k, num_particles)``.

    Falls back to the nearest particle count for the same ``k`` (in log-P
    distance), then to :data:`DEFAULT_PF_KDE_BANDWIDTH`.
    """
    if (k, num_particles) in RECOMMENDED_PF_KDE_BANDWIDTH:
        return RECOMMENDED_PF_KDE_BANDWIDTH[(k, num_particles)]
    same_k = [(P, h) for (kk, P), h in RECOMMENDED_PF_KDE_BANDWIDTH.items() if kk == k]
    if same_k:
        target = float(np.log(num_particles))
        _, h_best = min(same_k, key=lambda PH: abs(np.log(PH[0]) - target))
        return h_best
    return DEFAULT_PF_KDE_BANDWIDTH


def resolve_pf_bandwidths(pf_bandwidth, k: int, pf_counts) -> dict[int, float]:
    """Resolve a user-supplied bandwidth spec to a ``{pf_count: h}`` mapping.

    ``pf_bandwidth`` may be:
      * ``None`` — use :func:`recommended_pf_bandwidth` per count;
      * a float — broadcast to all counts;
      * a mapping ``{pf_count: h}`` — used as-is, with the recommendation
        as fallback for missing keys.
    """
    if pf_bandwidth is None:
        return {int(n): recommended_pf_bandwidth(k, int(n)) for n in pf_counts}
    if isinstance(pf_bandwidth, dict):
        return {
            int(n): float(pf_bandwidth.get(int(n), recommended_pf_bandwidth(k, int(n))))
            for n in pf_counts
        }
    return {int(n): float(pf_bandwidth) for n in pf_counts}


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
    ) -> tuple[ParticleSet, float]:
        key, simulate_key = jax.random.split(key, 2)
        particle_set = particle_set.simulate(env, action, next_observation, simulate_key)

        key, resample_key = jax.random.split(key, 2)
        particle_set = jax.lax.cond(
            particle_set.neff() < self.num_particles * self.resample_factor,
            lambda: particle_set.systematic_resample(resample_key),
            lambda: particle_set,
        )

        return particle_set, particle_set.neff()

    @eqx.filter_jit
    def reset(
        self, env: Environment, state: State, observation: Observation, key: jax.Array
    ) -> ParticleSet:
        particles, timesteps = jax.vmap(env.reset)(jax.random.split(key, self.num_particles))

        features = jax.vmap(lambda p: p.to_flat_features())(particles)
        particles = jax.vmap(lambda f: state.from_flat_features(f))(features)
        weights = env.observation_weights(particles, timesteps.observation, observation)

        return ParticleSet(particles, weights)
