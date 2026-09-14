import equinox as eqx
import jax
import jax.numpy as jnp
from embedex.distributions.embedding import FlowEmbedding
from jumanji.env import Environment

from nbf.base_types import Observation, State
from nbf.filters.particle import ParticleSet


class NeuralBayesianFilter(eqx.Module):
    """Generate, propagate, featurize, re-embed.

    The belief is the embedding of a uniformly weighted set of forecast particles, each
    paired with a sufficient statistic of every observation-likelihood factor at that
    particle (`env.likelihood_features`); the encoder consumes the likelihood features
    stream by stream. No importance weight is formed and nothing is resampled; the
    analysis step is what the encoder and generator learn. The bootstrap move
    (`ParticleSet.simulate`) is the only step that sees the domain's dynamics. The scalar
    weights on the returned particle sets are kept as a diagnostic (ESS).
    """

    model: FlowEmbedding
    num_particles: int

    def __repr__(self) -> str:
        return f'NBF ({self.num_particles})'

    @eqx.filter_jit
    def sample(self, embedding: jax.Array, state: State, num_samples: int, key: jax.Array) -> State:
        return jax.vmap(state.from_flat_features)(
            self.model.generate(embedding, num_samples, key=key)
        )

    @eqx.filter_jit
    def step(
        self,
        env: Environment,
        particles: State,
        action: int,
        next_observation: Observation,
        key: jax.Array,
    ) -> tuple[ParticleSet, jax.Array]:
        # Returns the propagated set with its likelihood features and the simulate step's
        # uniform-weight fallback flag.
        n = jax.tree.leaves(particles)[0].shape[0]
        particle_set, simulation_fallback = ParticleSet(particles, jnp.full(n, 1.0 / n)).simulate(
            env, action, next_observation, key
        )
        features = env.likelihood_features(
            particle_set.particles, next_observation, particle_set.weights
        )
        return (
            ParticleSet(particle_set.particles, particle_set.weights, features),
            simulation_fallback,
        )

    @eqx.filter_jit
    def update_particles(
        self,
        env: Environment,
        embedding: jax.Array,
        state: State,
        observation: Observation,
        action: int,
        next_observation: Observation,
        key: jax.Array,
    ) -> tuple[ParticleSet, jax.Array, jax.Array, jax.Array]:
        del observation
        sample_key, step_key = jax.random.split(key, 2)

        particles = self.sample(embedding, state, self.num_particles, sample_key)
        particle_set, simulation_fallback = self.step(
            env, particles, action, next_observation, step_key
        )
        return particle_set, particle_set.neff(), jnp.array(False), simulation_fallback

    @eqx.filter_jit
    def encoder_input(self, particle_set: ParticleSet) -> tuple[jax.Array, jax.Array]:
        """(states, likelihood) the encoder embeds: the flat state features of the set next
        to its likelihood features."""
        if particle_set.features is None:
            raise ValueError('particle set carries no likelihood features')
        states = jax.vmap(lambda p: p.to_flat_features())(particle_set.particles)
        return states, particle_set.features

    @eqx.filter_jit
    def embed_particles(self, particle_set: ParticleSet) -> jax.Array:
        states, likelihood = self.encoder_input(particle_set)
        return self.model.embed(states, likelihood)[0]

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
        particle_set, neff, sampling_fallback, simulation_fallback = self.update_particles(
            env, embedding, state, observation, action, next_observation, key
        )
        embedding = self.embed_particles(particle_set)

        return embedding, neff, sampling_fallback, simulation_fallback

    @eqx.filter_jit
    def reset_particles(
        self, env: Environment, state: State, observation: Observation, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        keys = jax.random.split(key, self.num_particles)
        if getattr(env, 'has_prior_sampler', False):
            particles = jax.vmap(lambda k: env.sample_prior(state, k))(keys)
        else:
            particles, _ = jax.vmap(env.reset)(keys)
            flat = jax.vmap(lambda p: p.to_flat_features())(particles)
            particles = jax.vmap(lambda f: state.from_flat_features(f))(flat)
        weights, _ = env.state_observation_weights(particles, observation)
        features = env.likelihood_features(particles, observation, weights)
        return self.encoder_input(ParticleSet(particles, weights, features))

    @eqx.filter_jit
    def reset(
        self, env: Environment, state: State, observation: Observation, key: jax.Array
    ) -> jax.Array:
        states, likelihood = self.reset_particles(env, state, observation, key)
        return self.model.embed(states, likelihood)[0]
