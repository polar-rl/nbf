import equinox as eqx
import jax
import jax.numpy as jnp
from embedex.distributions.embedding import FlowEmbedding
from jumanji.env import Environment

from nbf.base_types import Observation, State
from nbf.filters.particle import ParticleSet


class NeuralBayesianFilter(eqx.Module):
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
    def compare_state_obs(
        self, env: Environment, particles: State, observation: Observation
    ) -> tuple[jax.Array, jax.Array]:
        observations = jax.vmap(lambda s: s.observation_from_state())(particles)
        obs_weights, discarded = env.observation_weights(particles, observations, observation)

        return jax.lax.cond(
            env.has_state_observation,
            lambda: (obs_weights, discarded),
            lambda: (jnp.full(self.num_particles, 1 / self.num_particles), jnp.array(False)),
        )

    @eqx.filter_jit
    def step(
        self,
        env: Environment,
        embedding: jax.Array,
        state: State,
        observation: Observation,
        action: int,
        next_observation: Observation,
        key: jax.Array,
    ) -> tuple[ParticleSet, jax.Array, jax.Array]:
        # Besides the stepped particle set, returns the uniform-fallback flags of the two
        # weighting passes: sampling (generated particles vs the current observation) and
        # simulation (propagated particles vs the next observation).
        sample_key, simulate_key = jax.random.split(key, 2)

        particles = self.sample(embedding, state, self.num_particles - 1, sample_key)
        particles = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y[jnp.newaxis]], axis=0), particles, state
        )
        weights, sampling_fallback = self.compare_state_obs(env, particles, observation)
        particle_set = ParticleSet(particles, weights)

        particle_set, simulation_fallback = particle_set.simulate(
            env, action, next_observation, simulate_key
        )

        return particle_set, sampling_fallback, simulation_fallback

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
        # Generate a new gold state from the embedding so we can add it to the particle set
        # for the update without cheating. `update_with_ground_truth` puts the gold state in
        # directly, which is useful during search when we have already have a guess of the
        # gold state that matches the observation sequence in the tree.

        key, sample_key, step_key, resample_key = jax.random.split(key, 4)
        state = jax.tree.map(lambda x: x[0], self.sample(embedding, state, 1, sample_key))

        particle_set, sampling_fallback, simulation_fallback = self.step(
            env, embedding, state, observation, action, next_observation, step_key
        )
        neff = particle_set.neff()

        particle_set = jax.lax.cond(
            neff < self.num_particles * 0.5,
            lambda: particle_set.systematic_resample(resample_key),
            lambda: particle_set,
        )
        features = jax.vmap(lambda p: p.to_flat_features())(particle_set.particles)

        embedding = self.model.embed(features, particle_set.weights)[0]

        return embedding, neff, sampling_fallback, simulation_fallback

    @eqx.filter_jit
    def update_with_ground_truth(
        self,
        env: Environment,
        embedding: jax.Array,
        state: State,
        observation: Observation,
        action: int,
        next_observation: Observation,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        # Same as `update()`, except we don't replace the gold state with a sample. This helps
        # because we already have a state from simulation that matches the observation sequence
        # in the tree. If the model is good, this only adds bias and we can revisit calling it.

        key, step_key, resample_key = jax.random.split(key, 3)
        particle_set, sampling_fallback, simulation_fallback = self.step(
            env, embedding, state, observation, action, next_observation, step_key
        )
        neff = particle_set.neff()

        particle_set = jax.lax.cond(
            neff < self.num_particles * 0.5,
            lambda: particle_set.systematic_resample(resample_key),
            lambda: particle_set,
        )
        features = jax.vmap(lambda p: p.to_flat_features())(particle_set.particles)

        embedding = self.model.embed(features, particle_set.weights)[0]

        return embedding, neff, sampling_fallback, simulation_fallback

    @eqx.filter_jit
    def reset(
        self, env: Environment, state: State, observation: Observation, key: jax.Array
    ) -> jax.Array:
        particles, _ = jax.vmap(env.reset)(jax.random.split(key, self.num_particles))

        features = jax.vmap(lambda p: p.to_flat_features())(particles)
        particles = jax.vmap(lambda f: state.from_flat_features(f))(features)
        observations = jax.vmap(lambda p: p.observation_from_state())(particles)
        weights, _ = env.observation_weights(particles, observations, observation)

        return self.model.embed(features, weights)[0]
