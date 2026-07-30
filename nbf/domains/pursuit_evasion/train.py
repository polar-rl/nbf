import os

import jax
import numpy as np

from nbf.domains.pursuit_evasion.config import parse_args
from nbf.domains.pursuit_evasion.env import normalize_positions
from nbf.domains.pursuit_evasion.generate_data import (
    data_dirname,
    generate_data,
    load_data,
)
from nbf.train import run_training


def main() -> None:
    cfg = parse_args()
    key = jax.random.PRNGKey(cfg.seed)
    data_path = cfg.data_path
    os.makedirs(cfg.save_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    num_evaders = cfg.data.num_evaders
    data_file = os.path.join(data_path, data_dirname(num_evaders))

    if os.path.exists(data_file):
        print(f'Loading cached training data from {data_file}')
        data = load_data(data_path, num_evaders)
    else:
        print('Generating training data ...')
        key, data_key = jax.random.split(key)
        generate_data(
            cfg.data,
            num_samples=cfg.training.num_samples,
            key=data_key,
            out_dir=data_file,
        )
        data = load_data(data_path, num_evaders)
        print(f'Training data saved to {data_file}')

    run_training(
        cfg,
        normalize_fn=normalize_positions,
        train_array=np.asarray(data['particles_tp1']),
        train_weights=np.asarray(data['weights_tp1']),
        key=key,
    )


if __name__ == '__main__':
    main()
