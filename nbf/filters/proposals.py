"""Proposal distributions for the NBF update.

An NBF update generates particles from the current embedding, moves each one to the
next step under a proposal q(x' | x, y'), and weights it by T(x' | x) Y(y' | x') / q,
which is the self-normalized importance-sampling form the consistency theorem
covers for any proposal with bounded weights. The proposal is the one place where a
domain's structure can enter the update; the filter itself is proposal-agnostic.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jumanji.env import Environment

from nbf.base_types import Observation, State
from nbf.filters.particle import ParticleSet

OPTIMAL_PROPOSAL_HOOKS = (
    'propagate_mean_array',
    'marginal_observation_log_prob',
    'conjugate_move',
    'observation_present',
    'replace_state_array',
)


class Proposal(eqx.Module):
    """Move a weighted particle set one step and weight it against the observation."""

    tag: str = eqx.field(static=True, default='')

    def propose(
        self,
        env: Environment,
        particles: State,
        prior_weights: jax.Array,
        action: int,
        next_observation: Observation,
        key: jax.Array,
    ) -> tuple[ParticleSet, jax.Array]:
        """Return the proposed weighted set and a flag for a uniform-weight fallback."""
        raise NotImplementedError


class BootstrapProposal(Proposal):
    """q = T: simulate the transition and weight by the observation likelihood."""

    tag: str = eqx.field(static=True, default='')

    def propose(self, env, particles, prior_weights, action, next_observation, key):
        return ParticleSet(particles, prior_weights).simulate(env, action, next_observation, key)


def optimal_proposal_particle_set(
    env: Environment,
    particles: State,
    prior_weights: jax.Array,
    next_observation: Observation,
    key: jax.Array,
) -> ParticleSet:
    """Locally optimal proposal for a Gaussian transition observed through a linear
    Gaussian channel on a subset of coordinates: weight each particle by the marginal
    likelihood of y' at its propagated mean, then redraw the observed coordinates from
    the exact conditional. Falls back to the bootstrap move when no observation is
    present at this step."""
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


class OptimalProposal(Proposal):
    """The locally optimal proposal of `optimal_proposal_particle_set`.

    Requires the env hooks in OPTIMAL_PROPOSAL_HOOKS (Lorenz-96 provides them)."""

    tag: str = eqx.field(static=True, default='-OP')

    def propose(self, env, particles, prior_weights, action, next_observation, key):
        missing = [h for h in OPTIMAL_PROPOSAL_HOOKS if not hasattr(env, h)]
        if missing:
            raise ValueError(f'OptimalProposal requires env hooks: {missing}')
        pset = optimal_proposal_particle_set(env, particles, prior_weights, next_observation, key)
        return pset, jnp.array(False)


PROPOSALS = {'bootstrap': BootstrapProposal, 'optimal': OptimalProposal}


def make_proposal(name: str) -> Proposal:
    try:
        return PROPOSALS[name]()
    except KeyError:
        raise ValueError(f'unknown proposal {name!r}; choose from {sorted(PROPOSALS)}') from None
