import argparse
import importlib
import json
import os

import jax

# UKF/GMF require f64 for numerical stability.
jax.config.update('jax_enable_x64', True)

from nbf.eval.base import load_embedding_model, parse_seeds  # noqa: E402

EXPERIMENT_MODULES = {
    '1': 'nbf.domains.pursuit_evasion.experiments.experiment_1',
    '2': 'nbf.domains.pursuit_evasion.experiments.experiment_2',
    '3': 'nbf.domains.pursuit_evasion.experiments.experiment_3',
    '4': 'nbf.domains.pursuit_evasion.experiments.experiment_4',
    '5': 'nbf.domains.pursuit_evasion.experiments.experiment_5',
}


def _parse_args():
    parser = argparse.ArgumentParser(description='Pursuit-evasion evaluation (E1..E5)')
    parser.add_argument(
        '--experiments',
        type=str,
        default='',
        help='Comma-separated experiment ids from {1,2,3,4,5}.',
    )
    parser.add_argument('--model_dir', type=str, required=False, default='')
    parser.add_argument(
        '--model_dirs_per_k',
        type=str,
        default='',
        help='Optional ``k:path`` mapping for E2 (comma-separated).',
    )
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--episode_length', type=int, default=15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--seeds',
        type=str,
        default='',
        help='Comma-separated seed list. Overrides --num_seeds and --seed when set.',
    )
    parser.add_argument(
        '--num_seeds',
        type=int,
        default=0,
        help='If >1, run with seeds = range(--seed, --seed + num_seeds).',
    )
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_evaders', type=int, default=None)
    parser.add_argument(
        '--visualize_only',
        action='store_true',
        help='Skip evaluation; regenerate PDFs/TeX from existing npz files in --output_dir.',
    )

    parser.add_argument('--e1_methods', type=str, default='')
    parser.add_argument('--e2_methods', type=str, default='')
    parser.add_argument('--e2_ks', type=str, default='')
    parser.add_argument('--e3_methods', type=str, default='')
    parser.add_argument('--e4_sigma_methods', type=str, default='')
    parser.add_argument(
        '--e4_sigmas',
        type=str,
        default='',
        help='Comma-separated absolute sigma values for E4 sweep (default 0.01,0.05,0.1,0.2,0.5).',
    )
    parser.add_argument('--e5_methods', type=str, default='')
    parser.add_argument('--e5_num_episodes', type=int, default=4)
    parser.add_argument('--e5_num_warmup', type=int, default=1)
    parser.add_argument('--e5_num_repeats', type=int, default=3)
    return parser.parse_args()


def _maybe_pull_data_config_defaults(args) -> None:
    if not args.model_dir:
        if args.num_evaders is None:
            args.num_evaders = 1
        return
    data_config_path = os.path.join(args.model_dir, 'data_config.json')
    if not os.path.exists(data_config_path):
        if args.num_evaders is None:
            args.num_evaders = 1
        return
    with open(data_config_path) as f:
        data_cfg = json.load(f)
    if args.num_evaders is None:
        args.num_evaders = data_cfg.get('num_evaders', 1)
        print(f'Using num_evaders={args.num_evaders} from training config')


def _parse_model_dirs_per_k(spec: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        k_str, path = part.split(':', 1)
        out[int(k_str)] = path
    return out


def main() -> None:
    args = _parse_args()
    _maybe_pull_data_config_defaults(args)

    if args.output_dir is None:
        args.output_dir = f'results/pursuit_evasion/k{args.num_evaders}/eval'
    os.makedirs(args.output_dir, exist_ok=True)

    args.model_dirs_per_k_dict = _parse_model_dirs_per_k(args.model_dirs_per_k)
    args.seeds_list = parse_seeds(args)
    if len(args.seeds_list) > 1:
        print(f'Running with {len(args.seeds_list)} seeds: {args.seeds_list}')

    requested = [x.strip() for x in args.experiments.split(',') if x.strip()]
    if not requested:
        print('No experiments requested. Pass --experiments=1,2,3,4,5 to run.')
        return

    if args.visualize_only:
        for exp_id in requested:
            if exp_id not in EXPERIMENT_MODULES:
                print(f'Unknown experiment id {exp_id!r}; skipping.')
                continue
            mod = importlib.import_module(EXPERIMENT_MODULES[exp_id])
            if not hasattr(mod, 'replot'):
                print(f'E{exp_id}: no replot() defined; skipping.')
                continue
            print(f'\n=== Replot E{exp_id} from {args.output_dir} ===')
            try:
                mod.replot(args)
            except FileNotFoundError as e:
                print(f'E{exp_id}: missing npz, skipping ({e})')
        return

    master_key = jax.random.PRNGKey(args.seeds_list[0])
    if not args.model_dir:
        raise SystemExit('--model_dir is required unless --visualize_only is set.')
    master_key, model_key = jax.random.split(master_key)
    embedding_model, _ = load_embedding_model(args.model_dir, model_key)
    print(f'Loaded embedding model from {args.model_dir}')

    for exp_id in requested:
        if exp_id not in EXPERIMENT_MODULES:
            print(f'Unknown experiment id {exp_id!r}; skipping.')
            continue
        mod = importlib.import_module(EXPERIMENT_MODULES[exp_id])
        master_key, exp_key = jax.random.split(master_key)
        print(f'\n=== Running E{exp_id} ===')
        mod.run(args, embedding_model, exp_key)


if __name__ == '__main__':
    main()
