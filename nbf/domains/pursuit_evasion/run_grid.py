"""Pursuit-evasion evaluation grid: one array task per (cell, subcell, seed) shard,
--merge to build the tables. Mirrors nbf.domains.lorenz96.run_grid; the shard and
summary machinery is nbf.eval.grid.

Cells:
  id     one policy per (training-mixture component, seed), shared by the k evaders,
         observation sigma drawn per episode by the env (the in-distribution table and
         the scaling-in-k table, one grid per k)
  ood    the named held-out policies (zero-shot in the policy family)
  sigma  a random-walk policy with the observation sigma pinned to each value of a
         sweep that runs past the training range (held-out likelihoods)
Every method is scored on the true joint position: NLL for native densities (nbf,
enkf, gmf), RMSE, CRPS, coverage and spread for all. Observations are shuffled at k > 1,
so the belief is symmetric under evader relabelling and the sample metrics are computed
after relabelling every sample to the truth (`nbf.eval.metrics.relabel_samples_to_truth`).
"""

import argparse
import json
import os
import zlib

import jax

from nbf.config import build_dataclass
from nbf.domains.pursuit_evasion.config import DataConfig
from nbf.domains.pursuit_evasion.eval.base import make_env_kn, override_sigma
from nbf.domains.pursuit_evasion.eval.dispatch import (
    PE_EXTRA_SAMPLERS,
    compute_method_samples,
    compute_method_samples_and_density,
)
from nbf.domains.pursuit_evasion.policies import (
    OOD_POLICIES,
    TRAIN_MIXTURE_COMPONENTS,
    sample_train_component_policy,
)
from nbf.eval.base import (
    cast_velocity_net_bf16,
    load_embedding_model,
    parse_method_spec,
    parse_seeds,
    rollout_episodes,
)
from nbf.eval.dispatch import NUM_REF_SAMPLES
from nbf.eval.grid import (
    enumerate_tasks,
    merge_cell,
    run_merge,
    score_methods,
    shard_path,
    write_shard,
)

SIGMAS = (0.01, 0.02, 0.05)  # the precise-range regime; nothing above it
DEFAULT_CHUNK_SIZE = 4


def _sigma_tag(sigma: float) -> str:
    return 'sig' + f'{sigma:g}'.replace('.', 'p')


# (subcell, policy source, pinned sigma or None)
CELLS = {
    'id': [(comp, ('component', comp), None) for comp in TRAIN_MIXTURE_COMPONENTS],
    'ood': [(name, ('ood', name), None) for name in OOD_POLICIES],
    'sigma': [(_sigma_tag(s), ('component', 'random_walk'), s) for s in SIGMAS],
}


def subcell_labels(cell: str) -> list[tuple[str, str]]:
    out = []
    for subcell, (source, name), sigma in CELLS[cell]:
        label = f'{source} {name}' + (f', sigma_obs {sigma:g}' if sigma is not None else '')
        out.append((subcell, label))
    return out


DEFAULT_METHODS = (
    'nbf_512,nbf_1024,nbf_2048,pf_8192,pf_32768,pf_reg_8192,pf_reg_32768,'
    'enkf_50,enkf_100,gmf_16,gmf_64'
)

# Methods with a native density, scored by the NLL of the true state.
DENSITY_KINDS = ('nbf', 'gmf', 'enkf')


def make_policy(source: str, name: str, key):
    if source == 'ood':
        return OOD_POLICIES[name]
    if source == 'component':
        return sample_train_component_policy(key, name)
    raise ValueError(f'unknown policy source {source!r}')


def run_subcell(
    args, k, sampling_model, density_model, cell, subcell, policy_src, sigma, methods, seed
):
    key = jax.random.PRNGKey(seed)
    pol_key, rollout_key, noise_key, method_key = jax.random.split(key, 4)
    source, name = policy_src
    policy = make_policy(
        source, name, jax.random.fold_in(pol_key, zlib.crc32(name.encode()) % (2**31))
    )
    env = make_env_kn(
        k,
        args.episode_length,
        (policy,) * k,
        num_sensors=args.num_sensors,
        obs_interval=args.obs_interval,
        sigma_obs_min=args.sigma_obs_min,
        sigma_obs_max=args.sigma_obs_max,
    )
    episodes = rollout_episodes(env, args.episode_length, args.num_episodes, rollout_key)
    if sigma is not None:
        state0_b, obs0_b, states_next_b, _, _ = episodes
        episodes = override_sigma(env, state0_b, obs0_b, states_next_b, sigma, noise_key)
    states_next_b = episodes[2]
    return score_methods(
        methods,
        tag=f'{cell}/{subcell} k={k} m={args.num_sensors} D={args.obs_interval} seed {seed}',
        key=method_key,
        env=env,
        episodes=episodes,
        truth=states_next_b.positions,
        relabel_perms=env.perm_indices if k > 1 else None,
        sampling_model=sampling_model,
        density_model=density_model,
        density_kinds=DENSITY_KINDS,
        compute_samples=compute_method_samples,
        compute_samples_and_density=compute_method_samples_and_density,
        extra_samplers=PE_EXTRA_SAMPLERS,
        num_ref_samples=args.num_ref_samples,
        chunk_size=args.chunk_size,
        timing_repeats=args.timing_repeats,
        t_min=args.t_min,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Pursuit-evasion evaluation grid (id/ood/sigma).')
    parser.add_argument('cell', nargs='?', choices=sorted(CELLS))
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--k', type=int, default=0, help='evader count; 0 = from data_config.json')
    parser.add_argument(
        '--num_sensors',
        type=int,
        default=0,
        help='range sensors; 0 = from data_config.json (1 for models trained before the field)',
    )
    parser.add_argument(
        '--obs_interval',
        type=int,
        default=0,
        help='ranges every D steps; 0 = from data_config.json (1 for older models)',
    )
    parser.add_argument(
        '--sigma_obs_min',
        type=float,
        default=0.0,
        help='per-episode noise range; 0 = from data_config',
    )
    parser.add_argument(
        '--sigma_obs_max',
        type=float,
        default=0.0,
        help='per-episode noise range; 0 = from data_config',
    )
    parser.add_argument('--methods', type=str, default=DEFAULT_METHODS)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--episode_length', type=int, default=0)
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4')
    parser.add_argument('--num_seeds', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_ref_samples', type=int, default=NUM_REF_SAMPLES)
    parser.add_argument('--timing_repeats', type=int, default=0)
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        '--t_min', type=int, default=0, help='drop the first t_min steps from summaries'
    )
    parser.add_argument(
        '--fm_num_steps', type=int, default=0, help='ODE steps for the sampling path only'
    )
    parser.add_argument('--bf16_sampling', action='store_true')
    parser.add_argument('--task_id', type=int, default=-1)
    parser.add_argument('--list_tasks', action='store_true')
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = enumerate_tasks(CELLS, parse_seeds(args))
    if args.list_tasks:
        for i, (cell, subcell, _, sigma, seed) in enumerate(tasks):
            print(f'{i}\t{cell}\t{subcell}\tsigma={sigma}\tseed={seed}')
        print(f'\n{len(tasks)} tasks -> #SBATCH --array=0-{len(tasks) - 1}')
        return
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    if args.merge:
        run_merge(
            args.output_dir,
            'PE grid',
            CELLS,
            subcell_labels,
            methods,
            parse_seeds(args),
            args.t_min,
        )
        return
    if args.task_id < 0 and args.cell is None:
        raise SystemExit('give a cell, or --task_id / --list_tasks / --merge')
    if args.task_id >= len(tasks):
        raise SystemExit(f'--task_id {args.task_id} out of range (0..{len(tasks) - 1})')
    seeds = parse_seeds(args)

    k = args.k
    if args.model_dir is None:
        needs_model = [m for m in methods if parse_method_spec(m)[0] == 'nbf']
        if needs_model:
            raise SystemExit(f'--model_dir is required for methods {needs_model}')
        if not k:
            raise SystemExit('--k is required without --model_dir')
        data_cfg = DataConfig(
            num_evaders=k, num_sensors=args.num_sensors or 1, obs_interval=args.obs_interval or 1
        )
        if args.sigma_obs_min:
            data_cfg.sigma_obs_min = args.sigma_obs_min
        if args.sigma_obs_max:
            data_cfg.sigma_obs_max = args.sigma_obs_max
        sampling_model = density_model = None
        print('Methods-only mode: no model')
    else:
        with open(os.path.join(args.model_dir, 'data_config.json')) as f:
            data_cfg = build_dataclass(DataConfig, json.load(f))
        k = k or data_cfg.num_evaders
        if k != data_cfg.num_evaders:
            raise SystemExit(f'--k {k} does not match the model (k={data_cfg.num_evaders})')
        if args.num_sensors and args.num_sensors != data_cfg.num_sensors:
            raise SystemExit(
                f'--num_sensors {args.num_sensors} does not match the model (m={data_cfg.num_sensors})'
            )
        if args.obs_interval and args.obs_interval != data_cfg.obs_interval:
            raise SystemExit(
                f'--obs_interval {args.obs_interval} does not match the model (D={data_cfg.obs_interval})'
            )
        density_model, _ = load_embedding_model(args.model_dir, jax.random.PRNGKey(0))
        if args.fm_num_steps:
            sampling_model, _ = load_embedding_model(
                args.model_dir, jax.random.PRNGKey(0), fm_num_steps=args.fm_num_steps
            )
        else:
            sampling_model = density_model
        if args.bf16_sampling:
            sampling_model = cast_velocity_net_bf16(sampling_model)
        print(f'Loaded embedding model from {args.model_dir} (k={k})')
    args.episode_length = args.episode_length or data_cfg.episode_length
    args.num_sensors = args.num_sensors or data_cfg.num_sensors
    args.obs_interval = args.obs_interval or data_cfg.obs_interval
    if args.model_dir is not None and (args.sigma_obs_min or args.sigma_obs_max):
        raise SystemExit('the noise range comes from the model; do not pass --sigma_obs_min/max')
    args.sigma_obs_min = data_cfg.sigma_obs_min
    args.sigma_obs_max = data_cfg.sigma_obs_max
    print(
        f'noise range [{args.sigma_obs_min}, {args.sigma_obs_max}], m={args.num_sensors}, D={args.obs_interval}'
    )

    if args.task_id >= 0:
        cell, subcell, policy_src, sigma, seed = tasks[args.task_id]
        print(
            f'Task {args.task_id}/{len(tasks) - 1}: {cell}/{subcell} k={k} m={args.num_sensors} '
            f'D={args.obs_interval} seed {seed}, '
            f'{args.num_episodes} episodes x T={args.episode_length}, methods={methods}'
        )
        per_method = run_subcell(
            args, k, sampling_model, density_model, cell, subcell, policy_src, sigma, methods, seed
        )
        path = shard_path(args.output_dir, cell, subcell, seed)
        write_shard(path, cell, subcell, seed, per_method)
        print(f'Saved {path}\nRun --merge once every task finishes to build {cell}.npz')
        return

    print(
        f'Cell {args.cell}: {len(CELLS[args.cell])} subcell(s) x {len(seeds)} seed(s), k={k}, '
        f'{args.num_episodes} episodes x T={args.episode_length}, methods={methods}'
    )
    results = {}
    for subcell, policy_src, sigma in CELLS[args.cell]:
        for seed in seeds:
            results[(subcell, seed)] = run_subcell(
                args,
                k,
                sampling_model,
                density_model,
                args.cell,
                subcell,
                policy_src,
                sigma,
                methods,
                seed,
            )
    merge_cell(
        args.output_dir,
        'PE grid',
        args.cell,
        subcell_labels(args.cell),
        results,
        methods,
        args.t_min,
    )


if __name__ == '__main__':
    main()
