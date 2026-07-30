import equinox as eqx
import jax
import jax.numpy as jnp
from jumanji.env import Environment

from nbf.base_types import Observation, State
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.particle import ParticleSet

_OP_HOOKS = (
    'propagate_mean_array',
    'marginal_observation_log_prob',
    'conjugate_move',
    'observation_present',
    'replace_state_array',
)


def optimal_proposal_particle_set(
    env: Environment,
    particles: State,
    prior_weights: jax.Array,
    next_observation: Observation,
    key: jax.Array,
) -> ParticleSet:
    means = jax.vmap(env.propagate_mean_array)(particles)
    log_w = env.marginal_observation_log_prob(means, next_observation)
    finite = jnp.isfinite(log_w)
    log_w = jnp.where(finite, log_w, -jnp.inf)

    def _optimal():
        w = prior_weights * jax.nn.softmax(log_w)
        w_sum = jnp.sum(w)
        w = jax.lax.cond(w_sum > 0.0, lambda: w / w_sum, lambda: jax.nn.softmax(log_w))
        return env.conjugate_move(means, next_observation, key), w

    def _blind():
        return means + env.sigma_dyn * jax.random.normal(key, means.shape), prior_weights

    moved, weights = jax.lax.cond(
        env.observation_present(next_observation) & jnp.any(finite), _optimal, _blind
    )
    return ParticleSet(env.replace_state_array(particles, moved), weights)


class OptimalProposalNeuralBayesianFilter(NeuralBayesianFilter):
    def __repr__(self) -> str:
        return f'NBF-OP ({self.num_particles})'

    @eqx.filter_jit
    def update(
        self,
        env: Environment,
        embedding: jax.Array,
        state: State,
        observation: Observation,
        action: int,
        next_observation: Observation,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        missing = [h for h in _OP_HOOKS if not hasattr(env, h)]
        if missing:
            raise ValueError(f'OptimalProposalNeuralBayesianFilter requires env hooks: {missing}')
        key, sample_key, gen_key, move_key, resample_key = jax.random.split(key, 5)
        state = jax.tree.map(lambda x: x[0], self.sample(embedding, state, 1, sample_key))
        particles = self.sample(embedding, state, self.num_particles - 1, gen_key)
        particles = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y[jnp.newaxis]], axis=0), particles, state
        )
        weights, sampling_fallback = self.compare_state_obs(env, particles, observation)

        particle_set = optimal_proposal_particle_set(
            env, particles, weights, next_observation, move_key
        )
        neff = particle_set.neff()
        particle_set = jax.lax.cond(
            neff < self.num_particles * 0.5,
            lambda: particle_set.systematic_resample(resample_key),
            lambda: particle_set,
        )
        features = jax.vmap(lambda p: p.to_flat_features())(particle_set.particles)
        embedding = self.model.embed(features, particle_set.weights)[0]

        return embedding, neff, sampling_fallback, jnp.array(False)
