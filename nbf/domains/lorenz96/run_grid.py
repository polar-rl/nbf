import argparse
import dataclasses
import json
import os
import zlib

import jax
import numpy as np  # noqa: E402

from nbf.config import build_dataclass  # noqa: E402
from nbf.domains.lorenz96.config import DataConfig  # noqa: E402
from nbf.domains.lorenz96.env import Lorenz96Env  # noqa: E402
from nbf.domains.lorenz96.eval_utils import (  # noqa: E402
    L96_EXTRA_SAMPLERS,
    compute_l96_gold_smc_samples,
    compute_method_samples,
    compute_method_samples_and_density,
    sample_l96_negatives_batch,
)
from nbf.eval._tex import fmt_val, latex_escape  # noqa: E402
from nbf.eval.base import (  # noqa: E402
    cast_velocity_net_bf16,
    load_embedding_model,
    parse_method_spec,
    parse_seeds,
    rollout_episodes,
)
from nbf.eval.dispatch import (  # noqa: E402
    GOLD_SMC_COUNT,
    NUM_NEGATIVES,
    NUM_REF_SAMPLES,
)
from nbf.eval.metrics import (  # noqa: E402
    aggregate_mean_sem,
    mmd2_multibw_per_step_chunked,
    rmse_per_step_batch,
)
from nbf.eval.timing import time_method_walltime  # noqa: E402

TRAIN_SIGMA_OBS = (0.5, 1.5)
EVAL_SIGMA_OBS = (1.0, 1.0)
DEFAULT_CHUNK_SIZE = 4

CELLS = {
    'l1': [
        ('f6', ((6.0, 6.0),), EVAL_SIGMA_OBS),
        ('f10', ((10.0, 10.0),), EVAL_SIGMA_OBS),
    ],
    'l2': [
        ('f8', ((8.0, 8.0),), EVAL_SIGMA_OBS),
    ],
    'l3': [(f'f8_s{s}', ((8.0, 8.0),), (s, s)) for s in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)],
}

DEFAULT_METHODS = (
    'nbf_256,nbf_1024,nbf_2048,'
    'pf_1024,pf_8192,pf_65536,'
    'pf_reg_1024,pf_reg_8192,pf_reg_65536,'
    'pf_rm_1024,pf_rm_8192,pf_rm_65536,'
    'enkf_20,enkf_50,enkf_100,'
    'pf_oracle'
)

DENSITY_KINDS = ('nbf', 'nbf_op', 'pf', 'pf_reg', 'pf_rm', 'enkf')


def make_cell_env(data_cfg, f_bands, sigma_obs, episode_length: int) -> Lorenz96Env:
    return Lorenz96Env(
        K=data_cfg.K,
        time_limit=episode_length,
        dt=data_cfg.dt,
        substeps=data_cfg.window_substeps,
        sigma_dyn=data_cfg.sigma_dyn,
        sigma_obs_min=sigma_obs[0],
        sigma_obs_max=sigma_obs[1],
        f_bands=f_bands,
        spinup_time=data_cfg.spinup_time,
        sigma_init=data_cfg.sigma_init,
        norm_mean=data_cfg.norm_mean,
        norm_std=data_cfg.norm_std,
    )


def _atomic_savez(path: str, **arrays) -> None:
    """Write an npz via a temp file + rename, so a killed task can't leave a torn file.

    Array tasks are requeueable, and a partially written cache or shard would be
    silently loaded by whatever runs next.
    """
    # The .npz suffix is required: np.savez appends it when the name lacks one,
    # which would leave the rename pointing at a file that was never written.
    tmp = f'{path}.tmp{os.getpid()}.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def enumerate_tasks(seeds) -> list[tuple[str, str, tuple, tuple, int]]:
    tasks = []
    for cell in sorted(CELLS):
        for subcell, f_bands, sigma_obs in CELLS[cell]:
            for seed in seeds:
                tasks.append((cell, subcell, f_bands, sigma_obs, seed))
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
    _atomic_savez(path, **payload)


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


def oracle_cache_path(
    output_dir, cell, subcell, num_episodes, episode_length, S, seed, substeps, sigma_obs
) -> str:
    sig = f'{sigma_obs[0]:g}-{sigma_obs[1]:g}'
    fname = (
        f'l96_{cell}_{subcell}_N{num_episodes}_T{episode_length}_S{S}'
        f'_sub{substeps}_sig{sig}_seed{seed}.npz'
    )
    return os.path.join(output_dir, '_oracle_cache', fname)


def get_or_compute_l96_oracle(
    cache_path, env, state0_b, obs_next_b, num_ref_samples, key, chunk_size=0
):
    if os.path.exists(cache_path):
        with np.load(cache_path) as d:
            return jax.numpy.asarray(d['samples'])
    samples = compute_l96_gold_smc_samples(
        env,
        GOLD_SMC_COUNT,
        num_ref_samples,
        state0_b,
        obs_next_b,
        key,
        chunk_size=chunk_size,
        desc='gold-SMC oracle',
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    _atomic_savez(cache_path, samples=np.asarray(samples))
    return samples


def run_subcell(
    args, data_cfg, sampling_model, density_model, cell, subcell, f_bands, sigma_obs, methods, seed
):
    env = make_cell_env(data_cfg, f_bands, sigma_obs, args.episode_length)
    key = jax.random.PRNGKey(seed)
    rollout_key, oracle_key, oracle2_key, neg_key = jax.random.split(key, 4)
    state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = rollout_episodes(
        env, args.episode_length, args.num_episodes, rollout_key
    )
    cache = oracle_cache_path(
        args.output_dir,
        cell,
        subcell,
        args.num_episodes,
        args.episode_length,
        args.num_ref_samples,
        seed,
        data_cfg.window_substeps,
        sigma_obs,
    )
    oracle = get_or_compute_l96_oracle(
        cache, env, state0_b, obs_next_b, args.num_ref_samples, oracle_key, args.chunk_size
    )
    N, T = args.num_episodes, args.episode_length
    neg_pts_b = sample_l96_negatives_batch(
        env, jax.random.split(neg_key, N * T).reshape(N, T, -1), args.num_negatives
    )

    out = {}
    for spec in methods:
        kind, _ = parse_method_spec(spec)
        # crc32, not hash(): str hashing is salted per process, so array tasks and
        # requeued attempts would each draw different method keys.
        method_key = jax.random.fold_in(key, zlib.crc32(spec.encode()) % (2**31))
        density = None
        if spec == 'pf_oracle':
            samples = compute_l96_gold_smc_samples(
                env,
                GOLD_SMC_COUNT,
                args.num_ref_samples,
                state0_b,
                obs_next_b,
                oracle2_key,
                chunk_size=args.chunk_size,
                desc=f'{spec} samples',
            )
        elif kind in DENSITY_KINDS:
            samples, density = compute_method_samples_and_density(
                spec,
                env=env,
                embedding_model=sampling_model,
                density_model=density_model,
                k=data_cfg.K,
                state0_b=state0_b,
                obs0_b=obs0_b,
                states_next_b=states_next_b,
                obs_prev_b=obs_prev_b,
                obs_next_b=obs_next_b,
                neg_pts_b=neg_pts_b,
                num_ref_samples=args.num_ref_samples,
                key=method_key,
                chunk_size=args.chunk_size,
                desc=f'{spec} samples',
            )
        else:
            samples = compute_method_samples(
                spec,
                env=env,
                embedding_model=sampling_model,
                state0_b=state0_b,
                obs0_b=obs0_b,
                states_next_b=states_next_b,
                obs_prev_b=obs_prev_b,
                obs_next_b=obs_next_b,
                num_ref_samples=args.num_ref_samples,
                key=method_key,
                chunk_size=args.chunk_size,
                desc=f'{spec} samples',
            )

        res = {
            'mmd': np.asarray(
                mmd2_multibw_per_step_chunked(
                    samples, oracle, chunk_size=args.chunk_size, desc=f'{spec} mmd'
                )
            ),
            'rmse': np.asarray(rmse_per_step_batch(samples, states_next_b.x)),
        }
        if density is not None:
            res.update({name: np.asarray(v) for name, v in density.items()})
        if args.timing_repeats > 0 and spec != 'pf_oracle':
            timing = time_method_walltime(
                spec,
                env=env,
                embedding_model=sampling_model,
                state0_b=state0_b,
                obs0_b=obs0_b,
                states_next_b=states_next_b,
                obs_prev_b=obs_prev_b,
                obs_next_b=obs_next_b,
                num_ref_samples=args.num_ref_samples,
                num_warmup=1,
                num_repeats=args.timing_repeats,
                key=jax.random.fold_in(method_key, 1),
                extra_samplers=L96_EXTRA_SAMPLERS,
            )
            res['walltime_s'] = np.asarray(timing['mean_s'])
        out[spec] = res
        mmd_mean, mmd_sem = aggregate_mean_sem(res['mmd'][:, args.t_min :])
        rmse_mean, _ = aggregate_mean_sem(res['rmse'][:, args.t_min :])
        print(
            f'  [{cell}/{subcell} seed {seed}] {spec}: '
            f'mmd2={mmd_mean:.5f}+/-{mmd_sem:.5f} rmse={rmse_mean:.3f}'
        )
    return out


def _pool(results, subcell, spec, metric):
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


def write_summaries(args, cell, subcells, methods, results):
    md_lines = [f'# L96 grid cell {cell.upper()} (t>={args.t_min})', '']
    tex_lines = []
    for subcell, _, sigma_obs in subcells:
        md_lines += [
            f'## {subcell} (sigma_obs {sigma_obs[0]}-{sigma_obs[1]})',
            '',
            '| method | MMD$^2$ | LDS | AUC | RMSE | s/run |',
            '|---|---|---|---|---|---|',
        ]
        tex_lines += [
            f'% {cell} {subcell}',
            r'\begin{tabular}{lccccc}',
            r'method & MMD$^2$ & LDS & AUC & RMSE & s/run \\',
            r'\hline',
        ]
        for spec in methods:
            row = {}
            for metric in ('mmd', 'log_density_score', 'auc', 'rmse', 'walltime_s'):
                pooled = _pool(results, subcell, spec, metric)
                if pooled is None:
                    row[metric] = float('nan'), float('nan')
                elif metric == 'walltime_s':
                    row[metric] = float(np.mean(pooled)), float('nan')
                else:
                    row[metric] = aggregate_mean_sem(pooled[:, args.t_min :])
            cells = [
                f'{fmt_val(row["mmd"][0])} ± {fmt_val(row["mmd"][1])}',
                fmt_val(row['log_density_score'][0]),
                fmt_val(row['auc'][0]),
                f'{fmt_val(row["rmse"][0])} ± {fmt_val(row["rmse"][1])}',
                fmt_val(row['walltime_s'][0]),
            ]
            md_lines.append(f'| {spec} | ' + ' | '.join(cells) + ' |')
            tex_lines.append(latex_escape(spec) + ' & ' + ' & '.join(cells) + r' \\')
        md_lines.append('')
        tex_lines += [r'\end{tabular}', '']

    md_path = os.path.join(args.output_dir, f'{cell}_summary.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    tex_path = os.path.join(args.output_dir, f'{cell}.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines))
    print(f'Saved {md_path} and {tex_path}')


def merge_cell(args, cell: str, results: dict, methods: list[str]) -> None:
    """Reduce the shards of one cell into its npz + summary tables."""
    out = {}
    for (subcell, seed), per_method in results.items():
        for spec, metrics in per_method.items():
            for metric, arr in metrics.items():
                out[f'{subcell}|s{seed}|{spec}|{metric}'] = arr
    npz_path = os.path.join(args.output_dir, f'{cell}.npz')
    _atomic_savez(npz_path, **out)
    print(f'Saved {npz_path}')
    write_summaries(args, cell, CELLS[cell], methods, results)


def run_merge(args) -> None:
    by_cell = load_shards(args.output_dir)
    if not by_cell:
        raise SystemExit(f'No shards found under {os.path.join(args.output_dir, "_shards")}')
    requested = [m.strip() for m in args.methods.split(',') if m.strip()]
    seeds = parse_seeds(args)
    for cell in sorted(by_cell):
        results = by_cell[cell]
        present = {spec for per_method in results.values() for spec in per_method}
        methods = [m for m in requested if m in present]
        methods += sorted(present - set(methods))
        expected = len(CELLS[cell]) * len(seeds)
        if len(results) != expected:
            missing = sorted({(sc, s) for sc, _, _ in CELLS[cell] for s in seeds} - set(results))
            print(f'WARNING: {cell} has {len(results)}/{expected} shards; missing {missing}')
        merge_cell(args, cell, results, methods)


def main() -> None:
    parser = argparse.ArgumentParser(description='L96 experiment grid runner (L1/L2/L3).')
    parser.add_argument('cell', nargs='?', choices=sorted(CELLS))
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--methods', type=str, default=DEFAULT_METHODS)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--episode_length', type=int, default=0)
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4')
    parser.add_argument('--num_seeds', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_ref_samples', type=int, default=NUM_REF_SAMPLES)
    parser.add_argument('--num_negatives', type=int, default=NUM_NEGATIVES)
    parser.add_argument('--timing_repeats', type=int, default=0)
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        '--t_min',
        type=int,
        default=0,
        help='drop the first t_min steps (reference-noise-limited) from summaries and prints; '
        'shards always store full per-step arrays, so this can be applied at --merge time',
    )
    parser.add_argument(
        '--fm_num_steps',
        type=int,
        default=0,
        help='override flow ODE steps for the sampling/filter path only '
        '(0 = keep model config; log_prob for LDS/NLL/AUC always uses the config value)',
    )
    parser.add_argument(
        '--bf16_sampling',
        action='store_true',
        help='cast the sampling velocity net to bf16 (GPU; log_prob stays f32)',
    )
    parser.add_argument(
        '--task_id',
        type=int,
        default=-1,
        help='run one (cell, subcell, seed) unit from --list_tasks and write a shard',
    )
    parser.add_argument(
        '--substeps',
        type=int,
        default=0,
        help='override window_substeps (assimilation window = substeps * dt); '
        '0 = keep the model/config value. Regime-screen knob.',
    )
    parser.add_argument(
        '--sigma_obs',
        type=float,
        default=None,
        help='pin eval sigma_obs to this value for every cell, overriding CELLS.',
    )
    parser.add_argument('--list_tasks', action='store_true')
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = enumerate_tasks(parse_seeds(args))
    if args.list_tasks:
        for i, (cell, subcell, _, sigma_obs, seed) in enumerate(tasks):
            print(f'{i}\t{cell}\t{subcell}\tsigma_obs={sigma_obs}\tseed={seed}')
        print(f'\n{len(tasks)} tasks -> #SBATCH --array=0-{len(tasks) - 1}')
        return
    if args.merge:
        run_merge(args)
        return
    if args.task_id < 0 and args.cell is None:
        raise SystemExit('give a cell, or --task_id / --list_tasks / --merge')
    if args.task_id >= len(tasks):
        raise SystemExit(f'--task_id {args.task_id} out of range (0..{len(tasks) - 1})')
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    seeds = parse_seeds(args)

    if args.model_dir is None:
        needs_model = [m for m in methods if parse_method_spec(m)[0] == 'nbf']
        if needs_model:
            raise SystemExit(f'--model_dir is required for methods {needs_model}')
        data_cfg = DataConfig(norm_mean=0.0, norm_std=1.0)
        sampling_model = density_model = None
        print('Methods-only mode: no model, DataConfig defaults')
    else:
        with open(os.path.join(args.model_dir, 'data_config.json')) as f:
            data_cfg = build_dataclass(DataConfig, json.load(f))
        if data_cfg.norm_mean is None or data_cfg.norm_std is None:
            raise ValueError('model_dir data_config.json lacks norm stats; retrain or fix the file')

        density_model, _ = load_embedding_model(args.model_dir, jax.random.PRNGKey(0))
        if args.fm_num_steps:
            sampling_model, _ = load_embedding_model(
                args.model_dir, jax.random.PRNGKey(0), fm_num_steps=args.fm_num_steps
            )
        else:
            sampling_model = density_model
        if args.bf16_sampling:
            sampling_model = cast_velocity_net_bf16(sampling_model)
        print(
            f'Loaded embedding model from {args.model_dir} '
            f'(sampling: fm_num_steps={args.fm_num_steps or "config"}, '
            f'bf16={args.bf16_sampling})'
        )

    if args.substeps:
        data_cfg = dataclasses.replace(data_cfg, window_substeps=args.substeps)
        print(f'Window override: substeps={args.substeps} (window={args.substeps * data_cfg.dt:g})')
    args.episode_length = args.episode_length or data_cfg.episode_length

    if args.task_id >= 0:
        cell, subcell, f_bands, sigma_obs, seed = tasks[args.task_id]
        if args.sigma_obs is not None:
            sigma_obs = (args.sigma_obs, args.sigma_obs)
        print(
            f'Task {args.task_id}/{len(tasks) - 1}: cell {cell} subcell {subcell} seed {seed}, '
            f'{args.num_episodes} episodes x T={args.episode_length}, methods={methods}'
        )
        per_method = run_subcell(
            args,
            data_cfg,
            sampling_model,
            density_model,
            cell,
            subcell,
            f_bands,
            sigma_obs,
            methods,
            seed,
        )
        path = shard_path(args.output_dir, cell, subcell, seed)
        write_shard(path, cell, subcell, seed, per_method)
        print(f'Saved {path}\nRun --merge once every task finishes to build {cell}.npz')
        return

    subcells = CELLS[args.cell]
    if args.sigma_obs is not None:
        subcells = [(sc, fb, (args.sigma_obs, args.sigma_obs)) for sc, fb, _ in subcells]
    print(
        f'Cell {args.cell}: {len(subcells)} subcell(s) x {len(seeds)} seed(s), '
        f'{args.num_episodes} episodes x T={args.episode_length}, methods={methods}'
    )

    results = {}
    for subcell, f_bands, sigma_obs in subcells:
        for seed in seeds:
            results[(subcell, seed)] = run_subcell(
                args,
                data_cfg,
                sampling_model,
                density_model,
                args.cell,
                subcell,
                f_bands,
                sigma_obs,
                methods,
                seed,
            )

    merge_cell(args, args.cell, results, methods)


if __name__ == '__main__':
    main()
