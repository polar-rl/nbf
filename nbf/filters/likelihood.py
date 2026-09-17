"""Per-particle likelihood features: the vector the encoder receives in place of a scalar
importance weight.

For a likelihood that factors over sites, Y(y | x) = prod_k Y_k(y | x), the encoder is
given a sufficient statistic of every factor rather than their product. Environments opt in
by defining `likelihood_feature_dim` and `likelihood_features(particles, observation)`;
the generic fallback is the scalar log-weight, max-shifted so the best particle sits at 0.
The scalar case is the d = 1 special case of the same construction.
"""

import jax
import jax.numpy as jnp

LOG_FEATURE_FLOOR = -30.0
RESIDUAL_CLIP = 20.0


def likelihood_feature_dim(env) -> int:
    return int(getattr(env, 'likelihood_feature_dim', 1))


def scalar_log_weight_features(env, particles, observation) -> jax.Array:
    """(n, 1) max-shifted log-weights from `env.observation_weights`, floored."""
    observations = jax.vmap(lambda p: p.observation_from_state())(particles)
    weights, _ = env.observation_weights(particles, observations, observation)
    logw = jnp.log(jnp.maximum(weights, 1e-300))
    return jnp.maximum(logw - jnp.max(logw), LOG_FEATURE_FLOOR)[:, None]


def likelihood_features(env, particles, observation) -> jax.Array:
    """(n, d) features of the observation likelihood at each particle."""
    if hasattr(env, 'likelihood_features'):
        return env.likelihood_features(particles, observation)
    return scalar_log_weight_features(env, particles, observation)
