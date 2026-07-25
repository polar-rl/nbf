import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from nbf.metrics import median_heuristic, mmd2_u, rbf_kernel

DENSITY_METRIC_NAMES = ('log_density_score', 'nll', 'auc')
_MULTIBW_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


def aggregate_mean_sem(arr_NT: np.ndarray) -> tuple[float, float]:
    if arr_NT.size == 0 or np.all(np.isnan(arr_NT)):
        return float('nan'), float('nan')
    finite = np.isfinite(arr_NT).all(axis=1)
    if not finite.any():
        return float('nan'), float('nan')
    sub = arr_NT[finite]
    per_ep = sub.mean(axis=1)
    n = per_ep.shape[0]
    if n <= 1:
        return float(per_ep.mean()), 0.0
    return float(per_ep.mean()), float(per_ep.std(ddof=1) / np.sqrt(n))


def aggregate_per_step_mean_sem(arr_NT: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if arr_NT.size == 0:
        T = arr_NT.shape[1] if arr_NT.ndim == 2 else 0
        return np.full(T, np.nan), np.zeros(T)
    finite = np.isfinite(arr_NT).all(axis=1)
    if not finite.any():
        return np.full(arr_NT.shape[1], np.nan), np.zeros(arr_NT.shape[1])
    sub = arr_NT[finite]
    mean = sub.mean(axis=0)
    if sub.shape[0] > 1:
        se = sub.std(axis=0, ddof=1) / np.sqrt(sub.shape[0])
    else:
        se = np.zeros_like(mean)
    return mean, se


def density_metrics_from_logps(logp_pos: Array, logp_neg: Array) -> dict[str, Array]:
    M = logp_neg.shape[-1]
    logdens = logp_pos - jax.scipy.special.logsumexp(logp_neg, axis=-1) + jnp.log(M)
    nll = -logp_pos
    auc = jnp.mean((logp_pos[..., None] > logp_neg).astype(jnp.float32), axis=-1)
    return {'log_density_score': logdens, 'nll': nll, 'auc': auc}


@eqx.filter_jit
def mmd2_per_step_batch(method_samples: Array, oracle_samples: Array) -> Array:

    def _step(m, o):
        m_flat = m.reshape(m.shape[0], -1)
        o_flat = o.reshape(o.shape[0], -1)
        sigma = median_heuristic(o_flat) + 1e-6
        return mmd2_u(m_flat, o_flat, rbf_kernel(sigma))

    def _per_episode(mo):
        m, o = mo
        return jax.vmap(_step)(m, o)

    return jax.lax.map(_per_episode, (method_samples, oracle_samples))


def _multibw_rbf_kernel(sigma: float, scales=_MULTIBW_SCALES):
    inv_factors = tuple(1.0 / (2.0 * (sigma * s) ** 2) for s in scales)

    def _k(x, y):
        diff = x[:, None, :] - y[None, :, :]
        d2 = jnp.sum(diff * diff, axis=-1)
        return sum(jnp.exp(-inv * d2) for inv in inv_factors)

    return _k


@eqx.filter_jit
def mmd2_multibw_per_step_batch(method_samples: Array, oracle_samples: Array) -> Array:

    def _step(m, o):
        m_flat = m.reshape(m.shape[0], -1)
        o_flat = o.reshape(o.shape[0], -1)
        sigma = median_heuristic(o_flat) + 1e-6
        return mmd2_u(m_flat, o_flat, _multibw_rbf_kernel(sigma))

    return jax.vmap(jax.vmap(_step))(method_samples, oracle_samples)


@eqx.filter_jit
def rmse_per_step_batch(method_samples: Array, truth_states: Array) -> Array:
    S = method_samples.shape[2]
    mean = method_samples.reshape(*method_samples.shape[:2], S, -1).mean(axis=2)
    truth = truth_states.reshape(*truth_states.shape[:2], -1)
    return jnp.sqrt(jnp.mean((mean - truth) ** 2, axis=-1))


METRIC_FNS = {
    'mmd': mmd2_per_step_batch,
    'mmd_multibw': mmd2_multibw_per_step_batch,
}
