import json
import os
from dataclasses import asdict

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from embedex.distributions.builder import create_model
from embedex.distributions.training import train as _train
from jax import Array


def _count_parameters(model):
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return sum(leaf.size for leaf in leaves)


def _plot_loss_curve(losses, *, log_interval, smoothing_window=16):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    smoothed = np.convolve(losses, np.ones(smoothing_window), 'valid') / smoothing_window
    x = np.arange(len(smoothed)) * log_interval
    ax.plot(x, smoothed)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.grid(True)
    return fig


def make_generate_batch(train_array, train_weights, normalize_fn) -> callable:
    train_array = np.asarray(train_array)
    train_weights = np.asarray(train_weights)
    num_snapshots = train_array.shape[0]
    max_samples = train_array.shape[1]

    def generate_batch(batch_size: int, num_samples: int, key: Array) -> tuple[Array, Array]:
        seed = int(jax.random.randint(key, (), 0, np.iinfo(np.int32).max))
        rng = np.random.default_rng(seed)
        snap_indices = rng.integers(0, num_snapshots, size=batch_size)
        row_indices = rng.integers(0, max_samples, size=(batch_size, num_samples))
        batch = train_array[snap_indices[:, None], row_indices]
        wts = train_weights[snap_indices[:, None], row_indices]
        wts = wts / (np.sum(wts, axis=-1, keepdims=True) + 1e-30)
        return (
            normalize_fn(jnp.asarray(batch, dtype=jnp.float32)),
            jnp.asarray(wts, dtype=jnp.float32),
        )

    return generate_batch


def run_training(
    cfg,
    *,
    normalize_fn,
    train_array,
    train_weights,
    key: Array,
    numbered_checkpoints: bool = False,
) -> None:
    save_path = cfg.save_path
    os.makedirs(save_path, exist_ok=True)

    print(
        f'Training data: {train_array.shape[0]} snapshots, '
        f'{train_array.shape[1]} samples each, dim={train_array.shape[2]}'
    )

    cfg.model.input_dim = int(train_array.shape[2])
    cfg.model.output_dim = int(train_array.shape[2])

    key, model_key = jax.random.split(key)
    model = create_model(cfg.model, dequantize=False, key=model_key)
    print(f'Model parameters: {_count_parameters(model):,}')

    if cfg.load_model is not None:
        print(f'Loading model from {cfg.load_model}')
        with open(cfg.load_model, 'rb') as f:
            model = eqx.tree_deserialise_leaves(f, model)
        return

    print('Training ...')
    key, train_key = jax.random.split(key)
    generate_batch = make_generate_batch(train_array, train_weights, normalize_fn)

    model_path = os.path.join(save_path, 'model.eqx')

    config_path = os.path.join(save_path, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(asdict(cfg.model), f, indent=2)
    print(f'Config saved to {config_path}')

    data_config_path = os.path.join(save_path, 'data_config.json')
    with open(data_config_path, 'w') as f:
        json.dump(asdict(cfg.data), f, indent=2)
    print(f'Data config saved to {data_config_path}')

    if os.path.exists(model_path):
        print(f'[resume] loading existing checkpoint {model_path} (LR schedule restarts)')
        with open(model_path, 'rb') as f:
            model = eqx.tree_deserialise_leaves(f, model)

    def checkpoint_fn(step, ckpt_model):
        with open(model_path, 'wb') as f:
            eqx.tree_serialise_leaves(f, ckpt_model)
        if numbered_checkpoints:
            numbered = os.path.join(save_path, f'model_{step:06d}.eqx')
            with open(numbered, 'wb') as f:
                eqx.tree_serialise_leaves(f, ckpt_model)
        print(f'[checkpoint @ step {step}] saved to {save_path}')

    model, losses, norms = _train(
        model,
        cfg.training,
        generate_batch,
        key=train_key,
        checkpoint_fn=checkpoint_fn,
    )

    fig = _plot_loss_curve(losses, log_interval=cfg.training.log_interval)
    fig.savefig(os.path.join(save_path, 'loss-curve.pdf'), bbox_inches='tight')
    plt.close(fig)

    with open(model_path, 'wb') as f:
        eqx.tree_serialise_leaves(f, model)
    print(f'Model saved to {model_path}')
