import argparse
import json
import os

import jax

jax.config.update('jax_enable_x64', True)

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
    mmd2_multibw_per_step_batch,
    rmse_per_step_batch,
)
from nbf.eval.timing import time_method_walltime  # noqa: E402

TRAIN_SIGMA_OBS = (0.5, 1.5)

CELLS = {
    'l1': [
        ('f6', ((6.0, 6.0),), TRAIN_SIGMA_OBS),
        ('f10', ((10.0, 10.0),), TRAIN_SIGMA_OBS),
    ],
    'l2': [
        ('f8', ((8.0, 8.0),), TRAIN_SIGMA_OBS),
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

DENSITY_KINDS = ('nbf', 'pf', 'pf_reg', 'pf_rm', 'enkf')


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


def oracle_cache_path(output_dir, cell, subcell, num_episodes, episode_length, S, seed) -> str:
    fname = f'l96_{cell}_{subcell}_N{num_episodes}_T{episode_length}_S{S}_seed{seed}.npz'
    return os.path.join(output_dir, '_oracle_cache', fname)


def get_or_compute_l96_oracle(cache_path, env, state0_b, obs_next_b, num_ref_samples, key):
    if os.path.exists(cache_path):
        with np.load(cache_path) as d:
            return jax.numpy.asarray(d['samples'])
    samples = compute_l96_gold_smc_samples(
        env, GOLD_SMC_COUNT, num_ref_samples, state0_b, obs_next_b, key
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, samples=np.asarray(samples))
    return samples


def run_subcell(args, data_cfg, embedding_model, cell, subcell, f_bands, sigma_obs, methods, seed):
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
    )
    oracle = get_or_compute_l96_oracle(
        cache, env, state0_b, obs_next_b, args.num_ref_samples, oracle_key
    )
    N, T = args.num_episodes, args.episode_length
    neg_pts_b = sample_l96_negatives_batch(
        env, jax.random.split(neg_key, N * T).reshape(N, T, -1), args.num_negatives
    )

    out = {}
    for spec in methods:
        kind, _ = parse_method_spec(spec)
        method_key = jax.random.fold_in(key, hash(spec) % (2**31))
        density = None
        if spec == 'pf_oracle':
            samples = compute_l96_gold_smc_samples(
                env, GOLD_SMC_COUNT, args.num_ref_samples, state0_b, obs_next_b, oracle2_key
            )
        elif kind in DENSITY_KINDS:
            samples, density = compute_method_samples_and_density(
                spec,
                env=env,
                embedding_model=embedding_model,
                k=data_cfg.K,
                state0_b=state0_b,
                obs0_b=obs0_b,
                states_next_b=states_next_b,
                obs_prev_b=obs_prev_b,
                obs_next_b=obs_next_b,
                neg_pts_b=neg_pts_b,
                num_ref_samples=args.num_ref_samples,
                key=method_key,
            )
        else:
            samples = compute_method_samples(
                spec,
                env=env,
                embedding_model=embedding_model,
                state0_b=state0_b,
                obs0_b=obs0_b,
                states_next_b=states_next_b,
                obs_prev_b=obs_prev_b,
                obs_next_b=obs_next_b,
                num_ref_samples=args.num_ref_samples,
                key=method_key,
            )

        res = {
            'mmd': np.asarray(mmd2_multibw_per_step_batch(samples, oracle)),
            'rmse': np.asarray(rmse_per_step_batch(samples, states_next_b.x)),
        }
        if density is not None:
            res.update({name: np.asarray(v) for name, v in density.items()})
        if args.timing_repeats > 0 and spec != 'pf_oracle':
            timing = time_method_walltime(
                spec,
                env=env,
                embedding_model=embedding_model,
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
        mmd_mean, mmd_sem = aggregate_mean_sem(res['mmd'])
        rmse_mean, _ = aggregate_mean_sem(res['rmse'])
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
    md_lines = [f'# L96 grid cell {cell.upper()}', '']
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
                    row[metric] = aggregate_mean_sem(pooled)
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


def main() -> None:
    parser = argparse.ArgumentParser(description='L96 experiment grid runner (L1/L2/L3).')
    parser.add_argument('cell', choices=sorted(CELLS))
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--methods', type=str, default=DEFAULT_METHODS)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--episode_length', type=int, default=0)
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4')
    parser.add_argument('--num_seeds', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_ref_samples', type=int, default=NUM_REF_SAMPLES)
    parser.add_argument('--num_negatives', type=int, default=NUM_NEGATIVES)
    parser.add_argument('--timing_repeats', type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.model_dir, 'data_config.json')) as f:
        data_cfg = build_dataclass(DataConfig, json.load(f))
    if data_cfg.norm_mean is None or data_cfg.norm_std is None:
        raise ValueError('model_dir data_config.json lacks norm stats; retrain or fix the file')
    args.episode_length = args.episode_length or data_cfg.episode_length

    embedding_model, _ = load_embedding_model(args.model_dir, jax.random.PRNGKey(0))
    print(f'Loaded embedding model from {args.model_dir}')

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    seeds = parse_seeds(args)
    subcells = CELLS[args.cell]
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
                embedding_model,
                args.cell,
                subcell,
                f_bands,
                sigma_obs,
                methods,
                seed,
            )

    out = {}
    for (subcell, seed), per_method in results.items():
        for spec, metrics in per_method.items():
            for metric, arr in metrics.items():
                out[f'{subcell}|s{seed}|{spec}|{metric}'] = arr
    npz_path = os.path.join(args.output_dir, f'{args.cell}.npz')
    np.savez(npz_path, **out)
    print(f'Saved {npz_path}')

    write_summaries(args, args.cell, subcells, methods, results)


if __name__ == '__main__':
    main()
