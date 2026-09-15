"""Shared machinery for the per-domain evaluation grids (lorenz96.run_grid,
pursuit_evasion.run_grid): one array task per (cell, subcell, seed) writes a shard of
per-step metric arrays, and --merge reduces the shards into npz files and summary
tables. Domains own their CELLS and run_subcell; everything else lives here."""

import os
import zlib

import jax
import numpy as np

from nbf.eval._tex import fmt_val, latex_escape
from nbf.eval.base import parse_method_spec
from nbf.eval.metrics import aggregate_mean_sem, sample_metrics_per_step
from nbf.eval.timing import time_method_walltime

# Column order of the summary tables; every metric is an (N, T) per-step array in the
# shard except walltime_s (a scalar per method).
SUMMARY_METRICS = ('nll', 'rmse', 'crps', 'cov50', 'cov90', 'spread', 'walltime_s')
SUMMARY_HEADER_MD = '| method | NLL | RMSE | CRPS | cov50 | cov90 | spread/RMSE | s/run |'
SUMMARY_HEADER_TEX = r'method & NLL & RMSE & CRPS & cov50 & cov90 & spread/RMSE & s/run \\'


def atomic_savez(path: str, **arrays) -> None:
    """Write an npz via a temp file + rename, so a killed task can't leave a torn file.

    Array tasks are requeueable, and a partially written shard would be silently loaded
    by whatever runs next. The .npz suffix is required: np.savez appends it when the
    name lacks one, which would leave the rename pointing at a file never written.
    """
    tmp = f'{path}.tmp{os.getpid()}.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def enumerate_tasks(cells: dict, seeds) -> list[tuple]:
    """(cell, subcell, spec..., seed) for every subcell entry of every cell and seed."""
    tasks = []
    for cell in sorted(cells):
        for entry in cells[cell]:
            for seed in seeds:
                tasks.append((cell, *entry, seed))
    return tasks


def shard_path(output_dir: str, cell: str, subcell: str, seed: int) -> str:
    return os.path.join(output_dir, '_shards', f'{cell}__{subcell}__s{seed}.npz')


def write_shard(path: str, cell: str, subcell: str, seed: int, per_method: dict) -> None:
    payload = {
        f'{spec}|{metric}': arr for spec, m in per_method.items() for metric, arr in m.items()
    }
    payload['_cell'] = np.asarray(cell)
    payload['_subcell'] = np.asarray(subcell)
    payload['_seed'] = np.asarray(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic_savez(path, **payload)


def load_shards(output_dir: str) -> dict[str, dict]:
    """Read every shard back into ``{cell: {(subcell, seed): {spec: {metric: arr}}}}``."""
    by_cell: dict[str, dict] = {}
    shard_dir = os.path.join(output_dir, '_shards')
    if not os.path.isdir(shard_dir):
        return by_cell
    for fname in sorted(os.listdir(shard_dir)):
        if not fname.endswith('.npz'):
            continue
        with np.load(os.path.join(shard_dir, fname)) as d:
            cell = str(d['_cell'])
            subcell = str(d['_subcell'])
            seed = int(d['_seed'])
            per_method: dict[str, dict] = {}
            for key in d.files:
                if key.startswith('_'):
                    continue
                spec, _, metric = key.rpartition('|')
                per_method.setdefault(spec, {})[metric] = d[key]
        by_cell.setdefault(cell, {})[(subcell, seed)] = per_method
    return by_cell


def pool(results, subcell, spec, metric):
    """Concatenate one metric over every seed of a subcell (episodes stack on axis 0)."""
    arrs = [
        r[spec][metric]
        for (sc, _seed), r in results.items()
        if sc == subcell and spec in r and metric in r[spec]
    ]
    if not arrs:
        return None
    if metric == 'walltime_s':
        return np.asarray([float(a) for a in arrs])
    return np.concatenate(arrs, axis=0)


def summary_row(results, subcell, spec, t_min: int) -> dict[str, tuple[float, float]]:
    row = {}
    for metric in SUMMARY_METRICS:
        pooled = pool(results, subcell, spec, metric)
        if pooled is None:
            row[metric] = float('nan'), float('nan')
        elif metric == 'walltime_s':
            row[metric] = float(np.mean(pooled)), float('nan')
        else:
            row[metric] = aggregate_mean_sem(pooled[:, t_min:])
    return row


def format_row(row: dict) -> list[str]:
    return [
        f'{fmt_val(row["nll"][0])} ± {fmt_val(row["nll"][1])}',
        f'{fmt_val(row["rmse"][0])} ± {fmt_val(row["rmse"][1])}',
        f'{fmt_val(row["crps"][0])} ± {fmt_val(row["crps"][1])}',
        fmt_val(row['cov50'][0]),
        fmt_val(row['cov90'][0]),
        fmt_val(row['spread'][0] / row['rmse'][0]),
        fmt_val(row['walltime_s'][0]),
    ]


def write_summaries(output_dir, title, cell, subcell_labels, methods, results, t_min):
    """`subcell_labels` is [(subcell, label)] in table order."""
    md_lines = [f'# {title} cell {cell.upper()} (t>={t_min})', '']
    tex_lines = []
    for subcell, label in subcell_labels:
        md_lines += [f'## {subcell} ({label})', '', SUMMARY_HEADER_MD, '|---' * 8 + '|']
        tex_lines += [
            f'% {cell} {subcell}',
            r'\begin{tabular}{lccccccc}',
            SUMMARY_HEADER_TEX,
            r'\hline',
        ]
        for spec in methods:
            cells = format_row(summary_row(results, subcell, spec, t_min))
            md_lines.append(f'| {spec} | ' + ' | '.join(cells) + ' |')
            tex_lines.append(latex_escape(spec) + ' & ' + ' & '.join(cells) + r' \\')
        md_lines.append('')
        tex_lines += [r'\end{tabular}', '']

    md_path = os.path.join(output_dir, f'{cell}_summary.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    tex_path = os.path.join(output_dir, f'{cell}.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines))
    print(f'Saved {md_path} and {tex_path}')


def merge_cell(output_dir, title, cell, subcell_labels, results, methods, t_min) -> None:
    """Reduce the shards of one cell into its npz + summary tables."""
    out = {}
    for (subcell, seed), per_method in results.items():
        for spec, metrics in per_method.items():
            for metric, arr in metrics.items():
                out[f'{subcell}|s{seed}|{spec}|{metric}'] = arr
    npz_path = os.path.join(output_dir, f'{cell}.npz')
    atomic_savez(npz_path, **out)
    print(f'Saved {npz_path}')
    write_summaries(output_dir, title, cell, subcell_labels, methods, results, t_min)


def run_merge(output_dir, title, cells: dict, subcell_labels_of, requested_methods, seeds, t_min):
    """Merge every cell found under output_dir. `subcell_labels_of(cell)` gives the table
    order and labels; `requested_methods` fixes the row order, extras are appended."""
    by_cell = load_shards(output_dir)
    if not by_cell:
        raise SystemExit(f'No shards found under {os.path.join(output_dir, "_shards")}')
    for cell in sorted(by_cell):
        results = by_cell[cell]
        present = {spec for per_method in results.values() for spec in per_method}
        methods = [m for m in requested_methods if m in present]
        methods += sorted(present - set(methods))
        labels = subcell_labels_of(cell)
        expected = len(labels) * len(seeds)
        if len(results) != expected:
            missing = sorted({(sc, s) for sc, _ in labels for s in seeds} - set(results))
            print(f'WARNING: {cell} has {len(results)}/{expected} shards; missing {missing}')
        merge_cell(output_dir, title, cell, labels, results, methods, t_min)


def print_method_line(tag: str, spec: str, res: dict, t_min: int) -> None:
    rmse_mean, rmse_sem = aggregate_mean_sem(res['rmse'][:, t_min:])
    crps_mean, _ = aggregate_mean_sem(res['crps'][:, t_min:])
    nll_txt = ''
    if 'nll' in res:
        nll_mean, nll_sem = aggregate_mean_sem(res['nll'][:, t_min:])
        nll_txt = f' nll={nll_mean:.3f}+/-{nll_sem:.3f}'
    print(f'  [{tag}] {spec}:{nll_txt} rmse={rmse_mean:.3f}+/-{rmse_sem:.3f} crps={crps_mean:.3f}')


def score_methods(
    methods,
    *,
    tag: str,
    key,
    env,
    episodes,
    truth,
    sampling_model,
    density_model,
    density_kinds,
    compute_samples,
    compute_samples_and_density,
    extra_samplers,
    num_ref_samples: int,
    chunk_size: int,
    timing_repeats: int,
    t_min: int,
) -> dict:
    """Run every method on one batch of episodes and return {spec: {metric: (N, T)}}.

    `episodes` is (state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b) and
    `truth` the (N, T, *state) array the sample metrics score against. Methods whose
    kind is in `density_kinds` also get the NLL of the true state.
    """
    state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = episodes
    common = dict(
        env=env,
        embedding_model=sampling_model,
        state0_b=state0_b,
        obs0_b=obs0_b,
        states_next_b=states_next_b,
        obs_prev_b=obs_prev_b,
        obs_next_b=obs_next_b,
        num_ref_samples=num_ref_samples,
    )
    out = {}
    for spec in methods:
        kind, _ = parse_method_spec(spec)
        # crc32, not hash(): str hashing is salted per process, so array tasks and
        # requeued attempts would each draw different method keys.
        method_key = jax.random.fold_in(key, zlib.crc32(spec.encode()) % (2**31))
        density = None
        if kind in density_kinds:
            samples, density = compute_samples_and_density(
                spec,
                density_model=density_model,
                key=method_key,
                chunk_size=chunk_size,
                desc=f'{spec} samples',
                **common,
            )
        else:
            samples = compute_samples(
                spec, key=method_key, chunk_size=chunk_size, desc=f'{spec} samples', **common
            )
        res = sample_metrics_per_step(samples, truth)
        if density is not None:
            res.update({name: np.asarray(v) for name, v in density.items()})
        if timing_repeats > 0:
            timing = time_method_walltime(
                spec,
                num_warmup=1,
                num_repeats=timing_repeats,
                key=jax.random.fold_in(method_key, 1),
                extra_samplers=extra_samplers,
                **common,
            )
            res['walltime_s'] = np.asarray(timing['mean_s'])
        out[spec] = res
        print_method_line(tag, spec, res, t_min)
    return out
