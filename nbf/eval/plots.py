"""Figures from grid cell npz files (either domain): per-step curves, compute-matched
scatter (metric vs ms per filter step), and sweeps over subcells (sigma, OOD policies).

    python -m nbf.eval.plots per_step --npz grid/l2.npz --subcell f8 --out fig/l2_per_step.pdf
    python -m nbf.eval.plots compute  --npz grid/l2.npz --subcell f8 --metric crps --out fig/l2_compute.pdf
    python -m nbf.eval.plots sweep    --npz grid/sigma.npz --metric rmse --out fig/sigma.pdf
"""

import argparse
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nbf.eval.metrics import aggregate_mean_sem, aggregate_per_step_mean_sem  # noqa: E402

METRIC_LABELS = {
    'nll': 'NLL of the true state',
    'rmse': 'RMSE',
    'crps': 'CRPS',
    'cov50': '50% coverage',
    'cov90': '90% coverage',
    'spread': 'spread',
}

# family -> (label, colormap); intensity grows with the budget so a family reads as one hue
FAMILIES = {
    'nbf_op': ('NBF-OP', plt.get_cmap('Purples')),
    'nbf': ('NBF', plt.get_cmap('Blues')),
    'pf_reg': ('PF-reg', plt.get_cmap('Reds')),
    'pf_rm': ('PF-RM', plt.get_cmap('YlOrBr')),
    'pf_op': ('PF-OP', plt.get_cmap('RdPu')),
    'pf': ('PF', plt.get_cmap('Oranges')),
    'enkf': ('EnKF', plt.get_cmap('Greens')),
    'gmf': ('GMF', plt.get_cmap('Greens')),
}
_FAMILY_ORDER = list(FAMILIES)


def split_spec(spec: str) -> tuple[str, int]:
    family, _, n = spec.rpartition('_')
    return family, int(n)


def _intensity(n: int) -> float:
    lo, hi = 5.0, 16.0  # log2 budget range mapped to [0.45, 1.0]
    t = float(np.clip((np.log2(max(n, 1)) - lo) / (hi - lo), 0.0, 1.0))
    return 0.45 + 0.55 * t


def series_style(spec: str) -> tuple[str, object]:
    family, n = split_spec(spec)
    label, cmap = FAMILIES.get(family, (family, plt.get_cmap('Greys')))
    return f'{label}({n})', cmap(_intensity(n))


def order_specs(specs) -> list[str]:
    def key(spec):
        family, n = split_spec(spec)
        return (_FAMILY_ORDER.index(family) if family in _FAMILY_ORDER else 99, n)

    return sorted(specs, key=key)


def load_cell(npz_path: str) -> dict:
    """{subcell: {spec: {metric: (N_total, T)}}} with seeds pooled along axis 0."""
    out: dict = {}
    with np.load(npz_path) as d:
        for key in d.files:
            subcell, seed, spec, metric = key.split('|')
            out.setdefault(subcell, {}).setdefault(spec, {}).setdefault(metric, []).append(d[key])
    for subcell in out:
        for spec in out[subcell]:
            for metric, arrs in out[subcell][spec].items():
                if metric == 'walltime_s':
                    out[subcell][spec][metric] = np.asarray([float(a) for a in arrs])
                else:
                    out[subcell][spec][metric] = np.concatenate(arrs, axis=0)
    return out


def plot_per_step(cell, subcell, metrics, out_path, t_min=0):
    data = cell[subcell]
    specs = order_specs(data)
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.0), squeeze=False)
    for ax, metric in zip(axes[0], metrics, strict=True):
        for spec in specs:
            if metric not in data[spec]:
                continue
            mean, sem = aggregate_per_step_mean_sem(data[spec][metric])
            t = np.arange(1, mean.shape[0] + 1)
            label, color = series_style(spec)
            ax.plot(t[t_min:], mean[t_min:], color=color, label=label, lw=1.4)
            ax.fill_between(t[t_min:], (mean - sem)[t_min:], (mean + sem)[t_min:], color=color, alpha=0.15, lw=0)
        ax.set_xlabel('step t')
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.grid(True, alpha=0.3)
    axes[0][-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_compute_matched(cell, subcell, metric, out_path, t_min=0, num_episodes=None):
    """Metric (mean +/- SEM over episodes) against ms per filter step, one point per
    method, families joined by budget. walltime_s is the mean seconds for one full run of
    the subcell; ms/step needs the episode count and horizon of the shard."""
    data = cell[subcell]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    by_family: dict[str, list] = {}
    for spec in order_specs(data):
        d = data[spec]
        if metric not in d or 'walltime_s' not in d:
            continue
        N, T = d[metric].shape
        # walltime_s holds one mean-seconds value per shard (seed); the metric rows pool
        # every shard's episodes, so episodes per shard is N / number of shards.
        n_eps = num_episodes or N // len(d['walltime_s'])
        ms = 1000.0 * float(np.mean(d['walltime_s'])) / (n_eps * T)
        mean, sem = aggregate_mean_sem(d[metric][:, t_min:])
        family, _ = split_spec(spec)
        by_family.setdefault(family, []).append((ms, mean, sem, spec))
    for family, pts in by_family.items():
        pts.sort()
        xs, ys, es = zip(*[(p[0], p[1], p[2]) for p in pts], strict=True)
        label, cmap = FAMILIES.get(family, (family, plt.get_cmap('Greys')))
        ax.errorbar(xs, ys, yerr=es, color=cmap(0.8), marker='o', ms=4, lw=1.2, capsize=2, label=label)
    ax.set_xscale('log')
    ax.set_xlabel('ms per filter step')
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _subcell_value(subcell: str):
    m = re.fullmatch(r'sig(\d+)p?(\d*)', subcell)
    if m:
        return float(m.group(1) + ('.' + m.group(2) if m.group(2) else ''))
    m = re.fullmatch(r'f(\d+)', subcell)
    if m:
        return float(m.group(1))
    return None


def plot_sweep(cell, metric, out_path, t_min=0, xlabel=None):
    """Metric per method across the subcells of a cell: numeric subcells (sigma, F) as a
    line plot on a log x-axis when they span decades, categorical ones as grouped bars."""
    subcells = list(cell)
    values = [_subcell_value(sc) for sc in subcells]
    numeric = all(v is not None for v in values)
    specs = order_specs({s for sc in subcells for s in cell[sc]})
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    if numeric:
        order = np.argsort(values)
        xs = np.asarray(values)[order]
        for spec in specs:
            ys, es = [], []
            for i in order:
                d = cell[subcells[i]].get(spec, {})
                if metric in d:
                    m, s = aggregate_mean_sem(d[metric][:, t_min:])
                else:
                    m, s = np.nan, np.nan
                ys.append(m)
                es.append(s)
            label, color = series_style(spec)
            ax.errorbar(xs, ys, yerr=es, color=color, marker='o', ms=3.5, lw=1.2, capsize=2, label=label)
        if xs.max() / max(xs.min(), 1e-9) > 20:
            ax.set_xscale('log')
        ax.set_xlabel(xlabel or 'subcell value')
    else:
        width = 0.8 / max(len(specs), 1)
        x0 = np.arange(len(subcells))
        for j, spec in enumerate(specs):
            ys, es = [], []
            for sc in subcells:
                d = cell[sc].get(spec, {})
                m, s = aggregate_mean_sem(d[metric][:, t_min:]) if metric in d else (np.nan, np.nan)
                ys.append(m)
                es.append(s)
            label, color = series_style(spec)
            ax.bar(x0 + (j - len(specs) / 2 + 0.5) * width, ys, width, yerr=es, color=color, label=label, capsize=2)
        ax.set_xticks(x0)
        ax.set_xticklabels(subcells, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Figures from grid cell npz files.')
    parser.add_argument('kind', choices=['per_step', 'compute', 'sweep'])
    parser.add_argument('--npz', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--subcell', default=None, help='per_step / compute: which subcell')
    parser.add_argument('--metric', default='crps')
    parser.add_argument('--metrics', default='nll,rmse,crps', help='per_step: comma list')
    parser.add_argument('--t_min', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=None, help='compute: episodes per shard run')
    parser.add_argument('--xlabel', default=None)
    args = parser.parse_args()
    cell = load_cell(args.npz)
    if args.kind == 'per_step':
        subcell = args.subcell or next(iter(cell))
        plot_per_step(cell, subcell, args.metrics.split(','), args.out, args.t_min)
    elif args.kind == 'compute':
        subcell = args.subcell or next(iter(cell))
        plot_compute_matched(cell, subcell, args.metric, args.out, args.t_min, args.num_episodes)
    else:
        plot_sweep(cell, args.metric, args.out, args.t_min, args.xlabel)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
