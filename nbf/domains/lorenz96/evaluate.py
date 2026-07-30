import argparse
import json
import os

import jax

import numpy as np  # noqa: E402

from nbf.config import build_dataclass  # noqa: E402
from nbf.domains.lorenz96.config import DataConfig  # noqa: E402
from nbf.domains.lorenz96.env import build_lorenz96_env  # noqa: E402
from nbf.domains.lorenz96.eval_utils import compute_l96_gold_smc_samples  # noqa: E402
from nbf.eval.base import (  # noqa: E402
    cast_velocity_net_bf16,
    load_embedding_model,
    rollout_episodes,
)
from nbf.eval.dispatch import (  # noqa: E402
    GOLD_SMC_COUNT,
    NUM_REF_SAMPLES,
    compute_method_samples,
)
from nbf.eval.metrics import (  # noqa: E402
    aggregate_mean_sem,
    mmd2_multibw_per_step_chunked,
)

DEFAULT_CHUNK_SIZE = 4


def _parse_args():
    parser = argparse.ArgumentParser(description='Lorenz-96 evaluation (MMD vs gold-SMC oracle)')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--episode_length', type=int, default=0)
    parser.add_argument('--methods', type=str, default='nbf_1024,pf_8192')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        '--fm_num_steps',
        type=int,
        default=0,
        help='override flow ODE steps for sampling (0 = keep model config)',
    )
    parser.add_argument(
        '--bf16_sampling',
        action='store_true',
        help='cast the sampling velocity net to bf16 (GPU)',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.model_dir, 'data_config.json')) as f:
        data_cfg = build_dataclass(DataConfig, json.load(f))
    episode_length = args.episode_length or data_cfg.episode_length
    env = build_lorenz96_env(data_cfg, time_limit=episode_length)

    key = jax.random.PRNGKey(args.seed)
    key, model_key = jax.random.split(key)
    embedding_model, _ = load_embedding_model(
        args.model_dir, model_key, fm_num_steps=args.fm_num_steps or None
    )
    if args.bf16_sampling:
        embedding_model = cast_velocity_net_bf16(embedding_model)
    print(
        f'Loaded embedding model from {args.model_dir} '
        f'(fm_num_steps={args.fm_num_steps or "config"}, bf16={args.bf16_sampling})'
    )

    key, rollout_key, oracle_key = jax.random.split(key, 3)
    state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = rollout_episodes(
        env, episode_length, args.num_episodes, rollout_key
    )

    print(
        f'Computing gold-SMC oracle (N={GOLD_SMC_COUNT}, exact-redraw rejuvenation), '
        f'{args.num_episodes} episodes in chunks of {args.chunk_size or args.num_episodes} ...'
    )
    oracle = compute_l96_gold_smc_samples(
        env,
        GOLD_SMC_COUNT,
        NUM_REF_SAMPLES,
        state0_b,
        obs_next_b,
        oracle_key,
        chunk_size=args.chunk_size,
        desc='gold-SMC oracle',
    )

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    out = {'oracle_samples': np.asarray(oracle)}
    for spec in methods:
        key, method_key = jax.random.split(key)
        samples = compute_method_samples(
            spec,
            env=env,
            embedding_model=embedding_model,
            state0_b=state0_b,
            obs0_b=obs0_b,
            states_next_b=states_next_b,
            obs_prev_b=obs_prev_b,
            obs_next_b=obs_next_b,
            num_ref_samples=NUM_REF_SAMPLES,
            key=method_key,
            chunk_size=args.chunk_size,
            desc=f'{spec} samples',
        )
        mmd = np.asarray(
            mmd2_multibw_per_step_chunked(
                samples, oracle, chunk_size=args.chunk_size, desc=f'{spec} mmd'
            )
        )
        mean, sem = aggregate_mean_sem(mmd)
        print(f'{spec}: mmd2_multibw = {mean:.5f} +/- {sem:.5f}')
        out[f'mmd_{spec}'] = mmd

    out_path = os.path.join(args.output_dir, 'l96_mmd.npz')
    np.savez(out_path, **out)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
