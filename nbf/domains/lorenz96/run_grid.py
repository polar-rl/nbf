import argparse
import dataclasses
import json
import os

import jax

from nbf.config import build_dataclass
from nbf.domains.lorenz96.config import DataConfig
from nbf.domains.lorenz96.env import Lorenz96Env
from nbf.domains.lorenz96.eval_utils import (
    L96_EXTRA_SAMPLERS,
    compute_method_samples,
    compute_method_samples_and_density,
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

EVAL_SIGMA_OBS = (1.0, 1.0)
# Held-out likelihoods: the hard config trains on sigma_obs in [0.5, 1.1]; the sweep runs
# past both ends at the held-out forcing.
SIGMA_SWEEP = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
HELD_OUT_F = ((8.0, 8.0),)
DEFAULT_CHUNK_SIZE = 4


def _sigma_tag(sigma: float) -> str:
    return 'sig' + f'{sigma:g}'.replace('.', 'p')


# (subcell, forcing bands, (sigma_obs_min, sigma_obs_max)). Cells sort l1 < l2 < sigma,
# so the array-task indices of l1/l2 are unchanged by the sigma cell.
CELLS = {
    'l1': [
        ('f6', ((6.0, 6.0),), EVAL_SIGMA_OBS),
        ('f10', ((10.0, 10.0),), EVAL_SIGMA_OBS),
    ],
    'l2': [
        ('f8', HELD_OUT_F, EVAL_SIGMA_OBS),
    ],
    'sigma': [(_sigma_tag(sg), HELD_OUT_F, (sg, sg)) for sg in SIGMA_SWEEP],
}


def subcell_labels(cell: str) -> list[tuple[str, str]]:
    return [(sc, f'F {fb[0][0]:g}, sigma_obs {so[0]:g}') for sc, fb, so in CELLS[cell]]


DEFAULT_METHODS = (
    'nbf_256,nbf_1024,nbf_2048,'
    'pf_1024,pf_8192,pf_65536,'
    'pf_reg_1024,pf_reg_8192,pf_reg_65536,'
    'pf_rm_1024,pf_rm_8192,pf_rm_65536,'
    'enkf_20,enkf_50,enkf_100,'
    'letkf_20,letkf_50,letkf_100'
)

# Methods with a native density, scored by the NLL of the true state. Particle
# filters have none and get sample metrics only.
DENSITY_KINDS = ('nbf', 'enkf', 'letkf')


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


def run_subcell(
    args, data_cfg, sampling_model, density_model, cell, subcell, f_bands, sigma_obs, methods, seed
):
    env = make_cell_env(data_cfg, f_bands, sigma_obs, args.episode_length)
    key = jax.random.PRNGKey(seed)
    # Four-way split kept so episode and method keys match grids run before the
    # oracle and the negatives were removed.
    rollout_key, _, _, _ = jax.random.split(key, 4)
    state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = rollout_episodes(
        env, args.episode_length, args.num_episodes, rollout_key
    )

    return score_methods(
        methods,
        tag=f'{cell}/{subcell} seed {seed}',
        key=key,
        env=env,
        episodes=(state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b),
        truth=states_next_b.x,
        sampling_model=sampling_model,
        density_model=density_model,
        density_kinds=DENSITY_KINDS,
        compute_samples=compute_method_samples,
        compute_samples_and_density=compute_method_samples_and_density,
        extra_samplers=L96_EXTRA_SAMPLERS,
        num_ref_samples=args.num_ref_samples,
        chunk_size=args.chunk_size,
        timing_repeats=args.timing_repeats,
        t_min=args.t_min,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='L96 evaluation grid (l1: in-distribution F, l2: held-out F, sigma: held-out likelihoods).'
    )
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
    parser.add_argument('--timing_repeats', type=int, default=0)
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        '--t_min',
        type=int,
        default=0,
        help='drop the first t_min spin-up steps from summaries and prints; '
        'shards always store full per-step arrays, so this can be applied at --merge time',
    )
    parser.add_argument(
        '--fm_num_steps',
        type=int,
        default=0,
        help='override flow ODE steps for the sampling/filter path only '
        '(0 = keep model config; log_prob for the NLL always uses the config value)',
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

    tasks = enumerate_tasks(CELLS, parse_seeds(args))
    if args.list_tasks:
        for i, (cell, subcell, _, sigma_obs, seed) in enumerate(tasks):
            print(f'{i}\t{cell}\t{subcell}\tsigma_obs={sigma_obs}\tseed={seed}')
        print(f'\n{len(tasks)} tasks -> #SBATCH --array=0-{len(tasks) - 1}')
        return
    if args.merge:
        run_merge(
            args.output_dir,
            'L96 grid',
            CELLS,
            subcell_labels,
            [m.strip() for m in args.methods.split(',') if m.strip()],
            parse_seeds(args),
            args.t_min,
        )
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

    merge_cell(
        args.output_dir,
        'L96 grid',
        args.cell,
        subcell_labels(args.cell),
        results,
        methods,
        args.t_min,
    )


if __name__ == '__main__':
    main()
