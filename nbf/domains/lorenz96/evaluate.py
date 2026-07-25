import argparse
import json
import os

import jax

jax.config.update('jax_enable_x64', True)

import numpy as np  # noqa: E402

from nbf.config import build_dataclass  # noqa: E402
from nbf.domains.lorenz96.config import DataConfig  # noqa: E402
from nbf.domains.lorenz96.env import build_lorenz96_env  # noqa: E402
from nbf.domains.lorenz96.eval_utils import compute_l96_gold_smc_samples  # noqa: E402
from nbf.eval.base import load_embedding_model, rollout_episodes  # noqa: E402
from nbf.eval.dispatch import (  # noqa: E402
    GOLD_SMC_COUNT,
    NUM_REF_SAMPLES,
    compute_method_samples,
)
from nbf.eval.metrics import aggregate_mean_sem, mmd2_multibw_per_step_batch  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(description='Lorenz-96 evaluation (MMD vs gold-SMC oracle)')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--episode_length', type=int, default=0)
    parser.add_argument('--methods', type=str, default='nbf_1024,pf_8192')
    parser.add_argument('--seed', type=int, default=0)
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
    embedding_model, _ = load_embedding_model(args.model_dir, model_key)
    print(f'Loaded embedding model from {args.model_dir}')

    key, rollout_key, oracle_key = jax.random.split(key, 3)
    state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = rollout_episodes(
        env, episode_length, args.num_episodes, rollout_key
    )

    print(f'Computing gold-SMC oracle (N={GOLD_SMC_COUNT}, exact-redraw rejuvenation) ...')
    oracle = compute_l96_gold_smc_samples(
        env, GOLD_SMC_COUNT, NUM_REF_SAMPLES, state0_b, obs_next_b, oracle_key
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
        )
        mmd = np.asarray(mmd2_multibw_per_step_batch(samples, oracle))
        mean, sem = aggregate_mean_sem(mmd)
        print(f'{spec}: mmd2_multibw = {mean:.5f} +/- {sem:.5f}')
        out[f'mmd_{spec}'] = mmd

    out_path = os.path.join(args.output_dir, 'l96_mmd.npz')
    np.savez(out_path, **out)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
