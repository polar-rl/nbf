import argparse
import json
import os
import zlib

import jax
import numpy as np
from embedex.config import build_dataclass

from nbf.domains.pursuit_evasion.config import DataConfig
from nbf.domains.pursuit_evasion.eval.base import make_env_kn, override_sigma
from nbf.domains.pursuit_evasion.policies import (
    TRAIN_MIXTURE_COMPONENTS,
    sample_train_component_policy,
)
from nbf.eval.base import (
    batch_size,
    cast_velocity_net_bf16,
    chunked_stream,
    load_embedding_model,
    parse_seeds,
    rollout_episodes,
    split_T,
    time_len,
)
from nbf.eval.dispatch import NUM_REF_SAMPLES
from nbf.eval.grid import load_shards, merge_cell, shard_path, write_shard
from nbf.eval.metrics import (
    aggregate_mean_sem,
    density_metrics_from_logp,
    sample_metrics_per_step,
)
from nbf.eval.samples import nbf_crossover_stream_batch, pf_crossover_stream_batch
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.particle import ParticleFilter

CELL = 'mechanism'
DEFAULT_CHUNK_SIZE = 4


def method_names(n: int) -> list[str]:
    return [f'pf_{n}', f'pf_{n}_regen', f'nbf_{n}', f'nbf_{n}_resample']


def subcell_labels(sigma: float) -> list[tuple[str, str]]:
    return [(comp, f'component {comp}, sigma_obs {sigma:g}') for comp in TRAIN_MIXTURE_COMPONENTS]


def tasks_for(seeds) -> list[tuple[str, int]]:
    return [(comp, seed) for comp in TRAIN_MIXTURE_COMPONENTS for seed in seeds]


def run_task(args, k, sampling_model, density_model, component, seed):
    key = jax.random.PRNGKey(seed)
    pol_key, rollout_key, noise_key, pf_key, nbf_key = jax.random.split(key, 5)
    policy = sample_train_component_policy(
        jax.random.fold_in(pol_key, zlib.crc32(component.encode()) % (2**31)), component
    )
    env = make_env_kn(
        k,
        args.episode_length,
        (policy,) * k,
        num_sensors=args.num_sensors,
        obs_interval=args.obs_interval,
        sigma_obs_min=args.sigma_obs_min,
        sigma_obs_max=args.sigma_obs_max,
        feature_mode=args.feature_mode,
    )
    state0_b, obs0_b, states_next_b, _, _ = rollout_episodes(
        env, args.episode_length, args.num_episodes, rollout_key
    )
    episodes = override_sigma(env, state0_b, obs0_b, states_next_b, args.sigma, noise_key)
    state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = episodes
    truth = states_next_b.positions
    perms = env.perm_indices if k > 1 else None
    N = batch_size(state0_b)
    T = time_len(obs_prev_b)
    n = args.n
    nbf = NeuralBayesianFilter(sampling_model, num_particles=n)
    pf = ParticleFilter(num_particles=n)
    names = method_names(n)
    out = {}

    def _run(stream, key, desc):
        rk, sk, samp_k = jax.random.split(key, 3)
        return chunked_stream(
            stream,
            episodes,
            (jax.random.split(rk, N), split_T(sk, N, T), split_T(samp_k, N, T)),
            (args.num_ref_samples,),
            N,
            args.chunk_size,
            desc=desc,
        )

    def _score(samples, lp_true=None):
        res = sample_metrics_per_step(samples, truth, relabel_perms=perms)
        if lp_true is not None:
            res.update({m: np.asarray(v) for m, v in density_metrics_from_logp(lp_true).items()})
        return res

    pf_samples, regen, lp_regen, ess_pf = _run(
        lambda *a: pf_crossover_stream_batch(env, pf, nbf, density_model, *a), pf_key, 'pf loop'
    )
    out[names[0]] = _score(pf_samples)
    out[names[0]]['ess'] = np.asarray(ess_pf)
    out[names[1]] = _score(regen, lp_regen)

    nbf_samples, resampled, lp_nbf, ess_nbf = _run(
        lambda *a: nbf_crossover_stream_batch(env, nbf, density_model, *a), nbf_key, 'nbf loop'
    )
    out[names[2]] = _score(nbf_samples, lp_nbf)
    out[names[2]]['ess'] = np.asarray(ess_nbf)
    out[names[3]] = _score(resampled)

    tag = f'{component} k={k} sigma={args.sigma:g} n={n} seed {seed}'
    for name in names:
        r = out[name]
        crps, crps_sem = aggregate_mean_sem(r['crps'][:, args.t_min :])
        cov, _ = aggregate_mean_sem(r['cov50'][:, args.t_min :])
        nll = ''
        if 'nll' in r:
            m, s = aggregate_mean_sem(r['nll'][:, args.t_min :])
            nll = f' nll={m:.3f}+/-{s:.3f}'
        ess = ''
        if 'ess' in r:
            m, _ = aggregate_mean_sem(r['ess'][:, args.t_min :])
            ess = f' ess={m:.3f}'
        print(f'  [{tag}] {name}:{nll} crps={crps:.3f}+/-{crps_sem:.3f} cov50={cov:.3f}{ess}')
    return out


def write_pooled(output_dir, results, names, t_min):
    """Equal-weight average over the training-mixture policies (pooled episodes)."""
    lines = [
        f'# PE mechanism ablation, policy-averaged (t>={t_min})',
        '',
        '| row | NLL | CRPS | cov50 | cov90 | spread/RMSE | ESS/n |',
        '|---|---|---|---|---|---|---|',
    ]
    for name in names:
        cells = []
        for metric in ('nll', 'crps', 'cov50', 'cov90', 'spread', 'rmse', 'ess'):
            arrs = [r[name][metric] for r in results.values() if metric in r[name]]
            cells.append(aggregate_mean_sem(np.concatenate(arrs)[:, t_min:]) if arrs else None)
        nll, crps, cov50, cov90, spread, rmse, ess = cells
        fmt = lambda c, d=3: '--' if c is None else f'{c[0]:.{d}f}'  # noqa: E731
        pm = lambda c, d=3: '--' if c is None else f'{c[0]:.{d}f} ± {c[1]:.{d}f}'  # noqa: E731
        ratio = '--' if spread is None else f'{spread[0] / rmse[0]:.2f}'
        lines.append(
            f'| {name} | {pm(nll, 2)} | {pm(crps)} | {fmt(cov50, 2)} | {fmt(cov90, 2)} | {ratio} | {fmt(ess)} |'
        )
    path = os.path.join(output_dir, 'mechanism_pooled.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved {path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Pursuit-evasion mechanism ablation.')
    parser.add_argument('--model_dir', type=str, default=None, help='required for a task')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n', type=int, default=2048, help='particle budget of both filters')
    parser.add_argument('--sigma', type=float, default=0.02, help='pinned observation noise')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--episode_length', type=int, default=0)
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4')
    parser.add_argument('--num_seeds', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_ref_samples', type=int, default=NUM_REF_SAMPLES)
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument('--t_min', type=int, default=0)
    parser.add_argument('--fm_num_steps', type=int, default=0)
    parser.add_argument('--bf16_sampling', action='store_true')
    parser.add_argument('--task_id', type=int, default=-1)
    parser.add_argument('--list_tasks', action='store_true')
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seeds = parse_seeds(args)
    tasks = tasks_for(seeds)
    names = method_names(args.n)

    if args.list_tasks:
        for i, (comp, seed) in enumerate(tasks):
            print(f'{i}\t{comp}\tseed={seed}')
        print(f'\n{len(tasks)} tasks -> #SBATCH --array=0-{len(tasks) - 1}')
        return

    if args.merge:
        by_cell = load_shards(args.output_dir)
        if CELL not in by_cell:
            raise SystemExit(f'No shards under {os.path.join(args.output_dir, "_shards")}')
        results = by_cell[CELL]
        expected = len(TRAIN_MIXTURE_COMPONENTS) * len(seeds)
        if len(results) != expected:
            print(f'WARNING: {len(results)}/{expected} shards present')
        merge_cell(
            args.output_dir,
            'PE mechanism',
            CELL,
            subcell_labels(args.sigma),
            results,
            names,
            args.t_min,
        )
        write_pooled(args.output_dir, results, names, args.t_min)
        return

    if args.task_id < 0 or args.task_id >= len(tasks):
        raise SystemExit(f'--task_id in 0..{len(tasks) - 1}, or --list_tasks / --merge')
    if args.model_dir is None:
        raise SystemExit('--model_dir is required to run a task')

    with open(os.path.join(args.model_dir, 'data_config.json')) as f:
        data_cfg = build_dataclass(DataConfig, json.load(f))
    k = data_cfg.num_evaders
    density_model, _ = load_embedding_model(args.model_dir, jax.random.PRNGKey(0))
    if args.fm_num_steps:
        sampling_model, _ = load_embedding_model(
            args.model_dir, jax.random.PRNGKey(0), fm_num_steps=args.fm_num_steps
        )
    else:
        sampling_model = density_model
    if args.bf16_sampling:
        sampling_model = cast_velocity_net_bf16(sampling_model)
    args.episode_length = args.episode_length or data_cfg.episode_length
    args.num_sensors = data_cfg.num_sensors
    args.obs_interval = data_cfg.obs_interval
    args.sigma_obs_min = data_cfg.sigma_obs_min
    args.sigma_obs_max = data_cfg.sigma_obs_max
    args.feature_mode = data_cfg.feature_mode
    if not (args.sigma_obs_min <= args.sigma <= args.sigma_obs_max):
        raise SystemExit(
            f'--sigma {args.sigma} is outside the training range '
            f'[{args.sigma_obs_min}, {args.sigma_obs_max}]'
        )

    component, seed = tasks[args.task_id]
    print(
        f'Task {args.task_id}/{len(tasks) - 1}: {component} k={k} sigma={args.sigma:g} n={args.n} '
        f'seed {seed}, {args.num_episodes} episodes x T={args.episode_length}, '
        f'fm_steps={args.fm_num_steps or "model"} bf16={args.bf16_sampling}'
    )
    per_method = run_task(args, k, sampling_model, density_model, component, seed)
    path = shard_path(args.output_dir, CELL, component, seed)
    write_shard(path, CELL, component, seed, per_method)
    print(f'Saved {path}\nRun --merge once every task finishes')


if __name__ == '__main__':
    main()
