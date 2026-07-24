import os

import jax
import numpy as np

from nbf.domains.lorenz96.config import parse_args
from nbf.domains.lorenz96.generate_data import (
    data_dirname,
    generate_data,
    load_data,
    load_norm_stats,
)
from nbf.train import run_training


def main() -> None:
    cfg = parse_args()
    key = jax.random.PRNGKey(cfg.seed)
    os.makedirs(cfg.save_path, exist_ok=True)
    os.makedirs(cfg.data_path, exist_ok=True)

    data_file = os.path.join(cfg.data_path, data_dirname(cfg.data.K))
    if os.path.exists(data_file):
        print(f'Loading cached training data from {data_file}')
    else:
        print('Generating training data ...')
        key, data_key = jax.random.split(key)
        generate_data(
            cfg.data,
            num_samples=cfg.training.num_samples,
            key=data_key,
            out_dir=data_file,
        )
        print(f'Training data saved to {data_file}')
    data = load_data(cfg.data_path, cfg.data.K)

    stats = load_norm_stats(cfg.data_path, cfg.data.K)
    cfg.data.norm_mean = float(stats['norm_mean'])
    cfg.data.norm_std = float(stats['norm_std'])
    print(
        f'Climatological normalization: mean={cfg.data.norm_mean:.4f} std={cfg.data.norm_std:.4f}'
    )

    norm_mean, norm_std = cfg.data.norm_mean, cfg.data.norm_std

    def normalize_fn(arr):
        return (arr - norm_mean) / norm_std

    rows = cfg.data.episode_length + 1
    train_rows = (cfg.data.num_scenarios - cfg.data.num_val_scenarios) * rows
    train_array = np.asarray(data['particles_tp1'][:train_rows])
    train_weights = np.asarray(data['weights_tp1'][:train_rows])

    rng = np.random.default_rng(0)
    sample_rows = rng.integers(0, train_rows, size=min(train_rows, 512))
    sample = np.asarray(normalize_fn(train_array[sample_rows].astype(np.float32)))
    m, s = float(sample.mean()), float(sample.std())
    if not (abs(m) < 0.1 and 0.8 < s < 1.2):
        raise ValueError(f'Normalized-corpus sanity check failed: mean={m:.3f} std={s:.3f}')
    print(f'Normalized corpus sample: mean={m:.4f} std={s:.4f}')

    run_training(
        cfg,
        normalize_fn=normalize_fn,
        train_array=train_array,
        train_weights=train_weights,
        key=key,
        numbered_checkpoints=True,
    )


if __name__ == '__main__':
    main()
