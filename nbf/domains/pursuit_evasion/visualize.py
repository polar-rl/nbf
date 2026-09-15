import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from nbf.domains.pursuit_evasion.env import DEFAULT_OBSTACLES, L

_NBF_CMAP = plt.get_cmap('Blues')
_PF_CMAP = plt.get_cmap('Oranges')
_GMF_CMAP = plt.get_cmap('Greens')

_HEATMAP_CMAP = plt.cm.Blues.copy()
_HEATMAP_NORM = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=0.005)


def _draw_obstacles(ax, obstacles=DEFAULT_OBSTACLES):
    obstacles = np.asarray(obstacles)
    for obs in obstacles:
        rect = plt.Rectangle(
            (float(obs[0]), float(obs[1])),
            float(obs[2] - obs[0]),
            float(obs[3] - obs[1]),
            linewidth=1.5,
            edgecolor='black',
            facecolor='white',
            zorder=2,
        )
        ax.add_patch(rect)


def _logp_grid_to_pmf(grid_logp: np.ndarray) -> np.ndarray:
    g = np.asarray(grid_logp, dtype=np.float64)
    g = g - np.max(g)
    p = np.exp(g)
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        return np.zeros_like(p)
    return p / s


def plot_e1_heatmaps(
    all_grids: dict,
    true_positions: np.ndarray,
    timesteps,
    out_path: str,
    *,
    compact: bool = False,
    trajectory: np.ndarray | None = None,
):
    methods = list(all_grids.keys())
    timesteps = list(timesteps)
    n_rows = len(timesteps)
    n_cols = len(methods)

    if compact:
        cell = 2.6
        title_fs = 13
        tick_fs = 10
        ylabel_fs = 12
    else:
        cell = 2.6
        title_fs = 14
        tick_fs = 10
        ylabel_fs = 12

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(cell * n_cols, cell * n_rows),
        squeeze=False,
        layout='constrained',
    )

    for r, t in enumerate(timesteps):
        for c, spec in enumerate(methods):
            ax = axes[r][c]
            pmf = _logp_grid_to_pmf(all_grids[spec][r])
            ax.imshow(
                pmf,
                origin='lower',
                extent=[0, L, 0, L],
                cmap=_HEATMAP_CMAP,
                aspect='equal',
                norm=_HEATMAP_NORM,
            )
            _draw_obstacles(ax)
            if trajectory is not None:
                path = np.asarray(trajectory[: int(timesteps[r])])
                if path.shape[0] >= 2:
                    ax.plot(path[:, 0], path[:, 1], '-', color='red', linewidth=1.0, alpha=0.7)
            tp = np.asarray(true_positions[r])
            ax.plot(float(tp[0]), float(tp[1]), 'rx', markersize=8, mew=2.0)
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                label, _ = series_style(spec)
                ax.set_title(label, fontsize=title_fs)
            if c == 0:
                ax.set_ylabel(
                    f't={t}', fontsize=ylabel_fs, rotation=0, ha='right', va='center', labelpad=10
                )
            ax.tick_params(axis='both', labelsize=tick_fs)

    sm = plt.cm.ScalarMappable(cmap=_HEATMAP_CMAP, norm=_HEATMAP_NORM)
    cbar_ticks = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005]
    cbar = fig.colorbar(
        sm, ax=axes, location='bottom', shrink=0.45, pad=0.03, aspect=25, ticks=cbar_ticks
    )
    cbar.ax.set_xticklabels([f'{v:g}' for v in cbar_ticks])
    cbar.set_label('Probability density', fontsize=title_fs - 2)
    cbar.ax.tick_params(labelsize=tick_fs)

    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)


def _particle_intensity(p: int) -> float:
    lo, hi = 7.0, 15.0
    t = (np.log2(max(p, 1)) - lo) / (hi - lo)
    t = float(np.clip(t, 0.0, 1.0))
    return 0.4 + 0.6 * t


def series_style(name: str) -> tuple[str, object]:
    if name.startswith('nbf_'):
        try:
            p = int(name[len('nbf_') :])
        except ValueError:
            return (name, None)
        return (f'NBF({p})', _NBF_CMAP(_particle_intensity(p)))
    if name.startswith('pf_'):
        try:
            p = int(name.split('_', 1)[1])
        except ValueError:
            return (name, None)
        return (f'PF({p})', _PF_CMAP(_particle_intensity(p)))
    if name.startswith('gmf_'):
        try:
            m = int(name.split('_', 1)[1].lstrip('m').lstrip('M'))
        except ValueError:
            return (name, None)
        return (f'GMF({m})', _GMF_CMAP(0.4 + 0.15 * m))
    return (name, None)


def _series_sort_key(name: str) -> tuple[int, float, str]:
    if name.startswith('nbf_'):
        try:
            return (0, float(name.split('_', 1)[1]), name)
        except ValueError:
            return (0, 0.0, name)
    if name.startswith('pf_'):
        try:
            return (1, float(name.split('_', 1)[1]), name)
        except ValueError:
            return (1, 0.0, name)
    if name.startswith('gmf_'):
        try:
            return (2, float(name.split('_', 1)[1].lstrip('mM')), name)
        except ValueError:
            return (2, 0.0, name)
    return (9, 0.0, name)


def order_series(names) -> list[str]:
    filtered = [n for n in names if not n.startswith('gmf_')]
    return sorted(filtered, key=_series_sort_key)
