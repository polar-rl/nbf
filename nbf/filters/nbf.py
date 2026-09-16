import equinox as eqx
import jax
import jax.numpy as jnp
from embedex.distributions.embedding import FlowEmbedding
from jumanji.env import Environment

from nbf.base_types import Observation, State
from nbf.filters.particle import ParticleSet
from nbf.filters.proposals import BootstrapProposal, Proposal


class NeuralBayesianFilter(eqx.Module):
    """Generate, propose, reweight, re-embed. The proposal (bootstrap by default) is
    the only step that sees the domain's structure; see `nbf.filters.proposals`."""

    model: FlowEmbedding
    num_particles: int
    proposal: Proposal = eqx.field(default_factory=BootstrapProposal)

    def __repr__(self) -> str:
        return f'NBF{self.proposal.tag} ({self.num_particles})'

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
        # Besides the proposed particle set, returns the uniform-fallback flags of the two
        # weighting passes: sampling (generated particles vs the current observation) and
        # proposal (moved particles vs the next observation).
        sample_key, propose_key = jax.random.split(key, 2)

        particles = self.sample(embedding, state, self.num_particles - 1, sample_key)
        particles = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y[jnp.newaxis]], axis=0), particles, state
        )
        weights, sampling_fallback = self.compare_state_obs(env, particles, observation)
        particle_set, proposal_fallback = self.proposal.propose(
            env, particles, weights, action, next_observation, propose_key
        )

        return particle_set, sampling_fallback, proposal_fallback

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
        key, sample_key, step_key, resample_key = jax.random.split(key, 4)
        state = jax.tree.map(lambda x: x[0], self.sample(embedding, state, 1, sample_key))

        particle_set, sampling_fallback, simulation_fallback = self.step(
            env, embedding, state, observation, action, next_observation, step_key
        )
        neff = particle_set.neff()

        particle_set = self.maybe_resample(particle_set, resample_key)
        return particle_set, neff, sampling_fallback, simulation_fallback

    def maybe_resample(self, particle_set: ParticleSet, key: jax.Array) -> ParticleSet:
        """Systematic resample when the effective sample size drops below half."""
        return jax.lax.cond(
            particle_set.neff() < self.num_particles * 0.5,
            lambda: particle_set.systematic_resample(key),
            lambda: particle_set,
        )

    @eqx.filter_jit
    def embed_particles(self, particle_set: ParticleSet) -> jax.Array:
        features = jax.vmap(lambda p: p.to_flat_features())(particle_set.particles)
        return self.model.embed(features, particle_set.weights)[0]

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
        key, step_key, resample_key = jax.random.split(key, 3)
        particle_set, sampling_fallback, simulation_fallback = self.step(
            env, embedding, state, observation, action, next_observation, step_key
        )
        neff = particle_set.neff()

        particle_set = self.maybe_resample(particle_set, resample_key)
        features = jax.vmap(lambda p: p.to_flat_features())(particle_set.particles)

        embedding = self.model.embed(features, particle_set.weights)[0]

        return embedding, neff, sampling_fallback, simulation_fallback

    @eqx.filter_jit
    def reset_particles(
        self, env: Environment, state: State, observation: Observation, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        keys = jax.random.split(key, self.num_particles)
        if getattr(env, 'has_prior_sampler', False):
            particles = jax.vmap(lambda k: env.sample_prior(state, k))(keys)
            features = jax.vmap(lambda p: p.to_flat_features())(particles)
        else:
            particles, _ = jax.vmap(env.reset)(keys)
            features = jax.vmap(lambda p: p.to_flat_features())(particles)
            particles = jax.vmap(lambda f: state.from_flat_features(f))(features)
        observations = jax.vmap(lambda p: p.observation_from_state())(particles)
        weights, _ = env.observation_weights(particles, observations, observation)
        return features, weights

    @eqx.filter_jit
    def reset(
        self, env: Environment, state: State, observation: Observation, key: jax.Array
    ) -> jax.Array:
        features, weights = self.reset_particles(env, state, observation, key)
        return self.model.embed(features, weights)[0]
