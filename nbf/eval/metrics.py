import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


DENSITY_METRIC_NAMES = ('nll',)


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


def density_metrics_from_logp(logp_true: Array) -> dict[str, Array]:
    """Reference-free density score: the NLL of the true state under the method's belief.

    x_t is one exact draw from p(x_t | y_{1:t}), so the mean NLL over episodes is the
    posterior entropy plus the expected KL(p || q); differences between methods are
    differences in expected KL.
    """
    return {'nll': -logp_true}


@eqx.filter_jit
def rmse_per_step_batch(method_samples: Array, truth_states: Array) -> Array:
    S = method_samples.shape[2]
    mean = method_samples.reshape(*method_samples.shape[:2], S, -1).mean(axis=2)
    truth = truth_states.reshape(*truth_states.shape[:2], -1)
    return jnp.sqrt(jnp.mean((mean - truth) ** 2, axis=-1))


def _flatten_samples(method_samples: Array, truth_states: Array) -> tuple[Array, Array]:
    """(N, T, S, *state) -> (N, T, S, D) and (N, T, *state) -> (N, T, D)."""
    S = method_samples.shape[2]
    X = method_samples.reshape(*method_samples.shape[:2], S, -1)
    x = truth_states.reshape(*truth_states.shape[:2], -1)
    return X, x


@eqx.filter_jit
def crps_per_step_batch(method_samples: Array, truth_states: Array) -> Array:
    """Per-coordinate CRPS from an equal-weight sample set, averaged over coordinates.

    CRPS(F, x) = E|X - x| - 1/2 E|X - X'| with X, X' independent draws from the sample
    set. The second expectation uses the fair (distinct-pair) estimator
    2 / (S (S - 1)) sum_{i<j} |X_i - X_j|, computed from the sorted samples as
    sum_i (2 i - S - 1) X_(i) with 1-based i, so it is O(S log S) per coordinate and
    unbiased for a finite ensemble. Returns (N, T) in state units.
    """
    X, x = _flatten_samples(method_samples, truth_states)
    S = X.shape[2]
    term1 = jnp.mean(jnp.abs(X - x[:, :, None, :]), axis=2)  # (N, T, D)
    Xs = jnp.sort(X, axis=2)
    coef = 2.0 * jnp.arange(1, S + 1, dtype=X.dtype) - S - 1  # (S,)
    pair_sum = jnp.sum(coef[None, None, :, None] * Xs, axis=2)  # sum_{i<j} |X_i - X_j|
    term2 = pair_sum / (S * (S - 1))  # 1/2 * E|X - X'|
    return jnp.mean(term1 - term2, axis=-1)


@eqx.filter_jit
def coverage_per_step_batch(method_samples: Array, truth_states: Array, level: float) -> Array:
    """Fraction of coordinates whose marginal central `level` interval contains the truth.

    Intervals are per-coordinate empirical quantiles of the sample set at
    (1 - level) / 2 and (1 + level) / 2. Returns (N, T) in [0, 1]; a calibrated filter
    averages to `level`.
    """
    X, x = _flatten_samples(method_samples, truth_states)
    lo = jnp.quantile(X, (1.0 - level) / 2.0, axis=2)
    hi = jnp.quantile(X, (1.0 + level) / 2.0, axis=2)
    inside = (x >= lo) & (x <= hi)
    return jnp.mean(inside.astype(X.dtype), axis=-1)


@eqx.filter_jit
def spread_per_step_batch(method_samples: Array) -> Array:
    """Ensemble spread: sqrt of the coordinate-averaged sample variance, (N, T).

    Paired with rmse_per_step_batch it gives the spread-skill ratio; a calibrated
    ensemble has spread close to RMSE.
    """
    S = method_samples.shape[2]
    X = method_samples.reshape(*method_samples.shape[:2], S, -1)
    var = jnp.var(X, axis=2, ddof=1)
    return jnp.sqrt(jnp.mean(var, axis=-1))


COVERAGE_LEVELS = (0.5, 0.9)


def sample_metrics_per_step(method_samples: Array, truth_states: Array) -> dict[str, np.ndarray]:
    """Every reference-free sample metric as (N, T) numpy arrays, keyed by summary name."""
    out = {
        'rmse': np.asarray(rmse_per_step_batch(method_samples, truth_states)),
        'crps': np.asarray(crps_per_step_batch(method_samples, truth_states)),
        'spread': np.asarray(spread_per_step_batch(method_samples)),
    }
    for level in COVERAGE_LEVELS:
        out[f'cov{int(round(level * 100))}'] = np.asarray(
            coverage_per_step_batch(method_samples, truth_states, level)
        )
    return out
