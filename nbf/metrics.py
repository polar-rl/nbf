import equinox as eqx
import jax.numpy as jnp
from jax import Array


def _pairwise_sq_dists(x: Array, y: Array) -> Array:
    """Squared Euclidean distance matrix between rows of ``x`` (m, d) and ``y`` (n, d)."""
    diff = x[:, None, :] - y[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


def rbf_kernel(sigma: float):
    """Return ``k(x, y) = exp(-||x - y||^2 / (2 sigma^2))`` on row-stacked inputs."""
    inv = 1.0 / (2.0 * sigma * sigma)

    def _k(x: Array, y: Array) -> Array:
        return jnp.exp(-inv * _pairwise_sq_dists(x, y))

    return _k


def median_heuristic(ref: Array) -> Array:
    """Median of off-diagonal pairwise Euclidean distances within ``ref`` (n, d).

    The classic bandwidth heuristic for an RBF kernel: ``sigma = median(||x_i - x_j||)``
    over the strict upper triangle. Returns a scalar JAX array.
    """
    sq = _pairwise_sq_dists(ref, ref)
    n = ref.shape[0]
    iu = jnp.triu_indices(n, k=1)
    return jnp.median(jnp.sqrt(sq[iu]))


@eqx.filter_jit
def mmd2_u(x: Array, y: Array, kernel) -> Array:
    """Unbiased U-statistic estimator of ``MMD^2`` between samples ``x`` and ``y``.

    Both inputs must be row-stacked sample matrices ``(m, d)`` and ``(n, d)``. The
    estimator excludes diagonal terms in the within-sample sums, so it can be
    negative for finite ``m, n``.
    """
    m = x.shape[0]
    n = y.shape[0]
    Kxx = kernel(x, x)
    Kyy = kernel(y, y)
    Kxy = kernel(x, y)
    sum_xx = jnp.sum(Kxx) - jnp.trace(Kxx)
    sum_yy = jnp.sum(Kyy) - jnp.trace(Kyy)
    sum_xy = jnp.sum(Kxy)
    return sum_xx / (m * (m - 1)) + sum_yy / (n * (n - 1)) - 2.0 * sum_xy / (m * n)
