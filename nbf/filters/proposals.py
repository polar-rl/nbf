"""The locally optimal proposal move used by the PF-OP baseline.

NBF itself uses the bootstrap move (`ParticleSet.simulate`), so this module serves only
the particle-filter baselines that propose under q(x' | x, y') and weight by
T(x' | x) Y(y' | x') / q.
"""

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
