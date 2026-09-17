import equinox as eqx
import jax
import jax.numpy as jnp
from embedex.distributions.embedding import FlowEmbedding
from jumanji.env import Environment

from nbf.base_types import Observation, State
from nbf.filters.likelihood import likelihood_features
from nbf.filters.particle import ParticleSet
from nbf.filters.proposals import BootstrapProposal, Proposal


class NeuralBayesianFilter(eqx.Module):
    """Generate, propose, featurize, re-embed.

    The belief is the embedding of a uniformly weighted set whose elements are
    (state features, likelihood features): the forecast particles together with a
    sufficient statistic of every observation-likelihood factor at each particle
    (`nbf.filters.likelihood`). No importance weight is formed and nothing is resampled;
    the analysis step is what the encoder and generator learn. The proposal (bootstrap by
    default) is the only step that sees the domain's dynamics; see `nbf.filters.proposals`.
    The scalar weights on the returned particle sets are kept as a diagnostic (ESS).
    """

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
        """Likelihood weights of a set of states under an observation. Not used by the
        filter itself (the belief the particles are generated from already conditions on
        the current observation); kept for planners that weight rollouts from generated
        states (BETS)."""
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
        # `observation` (the current one) is already accounted for by the belief the
        # particles are generated from; it stays in the signature for callers. Returns the
        # proposed set with its likelihood features, a constant False (no sampling pass),
        # and the proposal's uniform-fallback flag.
        del observation
        sample_key, propose_key = jax.random.split(key, 2)

        particles = self.sample(embedding, state, self.num_particles - 1, sample_key)
        particles = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y[jnp.newaxis]], axis=0), particles, state
        )
        uniform = jnp.full(self.num_particles, 1.0 / self.num_particles)
        particle_set, proposal_fallback = self.proposal.propose(
            env, particles, uniform, action, next_observation, propose_key
        )
        features = likelihood_features(env, particle_set.particles, next_observation)
        particle_set = ParticleSet(particle_set.particles, particle_set.weights, features)
        return particle_set, jnp.array(False), proposal_fallback

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
        key, sample_key, step_key = jax.random.split(key, 3)
        state = jax.tree.map(lambda x: x[0], self.sample(embedding, state, 1, sample_key))

        particle_set, sampling_fallback, simulation_fallback = self.step(
            env, embedding, state, observation, action, next_observation, step_key
        )
        return particle_set, particle_set.neff(), sampling_fallback, simulation_fallback

    @eqx.filter_jit
    def encoder_input(self, particle_set: ParticleSet) -> tuple[jax.Array, jax.Array]:
        """(features, weights) the encoder embeds: state features concatenated with the
        likelihood features, under uniform weights."""
        if particle_set.features is None:
            raise ValueError('particle set carries no likelihood features')
        states = jax.vmap(lambda p: p.to_flat_features())(particle_set.particles)
        features = jnp.concatenate([states, particle_set.features], axis=-1)
        n = features.shape[0]
        return features, jnp.full(n, 1.0 / n)

    @eqx.filter_jit
    def embed_particles(self, particle_set: ParticleSet) -> jax.Array:
        features, weights = self.encoder_input(particle_set)
        return self.model.embed(features, weights)[0]

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
        particle_set, sampling_fallback, simulation_fallback = self.step(
            env, embedding, state, observation, action, next_observation, key
        )
        embedding = self.embed_particles(particle_set)
        return embedding, particle_set.neff(), sampling_fallback, simulation_fallback

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
        weights = jnp.full(self.num_particles, 1.0 / self.num_particles)
        features = likelihood_features(env, particles, observation)
        return self.encoder_input(ParticleSet(particles, weights, features))

    @eqx.filter_jit
    def reset(
        self, env: Environment, state: State, observation: Observation, key: jax.Array
    ) -> jax.Array:
        features, weights = self.reset_particles(env, state, observation, key)
        return self.model.embed(features, weights)[0]
