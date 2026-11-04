"""Per-particle likelihood features: the vector the encoder receives in place of a scalar
importance weight.

Every environment defines `likelihood_feature_dim` and
`likelihood_features(particles, observation, weights)`. For a likelihood that factors over
sites, Y(y | x) = prod_k Y_k(y | x), the features are a sufficient statistic of every factor
(Lorenz-96's per-site residuals, pursuit-evasion's per-sensor ones). A domain whose
observation model does not factor hands back `log_weight_features` of
the set's accumulated weights, the scalar case of the same construction.
"""

import jax
import jax.numpy as jnp

LOG_FEATURE_FLOOR = -30.0
RESIDUAL_CLIP = 20.0


def log_weight_features(weights: jax.Array) -> jax.Array:
    """(n, 1) max-shifted log of a particle set's weights, floored."""
    logw = jnp.log(jnp.maximum(weights, 1e-300))
    return jnp.maximum(logw - jnp.max(logw), LOG_FEATURE_FLOOR)[:, None]
