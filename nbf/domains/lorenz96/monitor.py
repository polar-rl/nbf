import argparse
import csv
import glob
import json
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from nbf.data import load_arrays  # noqa: E402
from nbf.domains.lorenz96.generate_data import data_dirname  # noqa: E402
from nbf.eval.base import load_embedding_model  # noqa: E402
from nbf.eval.metrics import mmd2_multibw_per_step_batch  # noqa: E402


def load_latest_checkpoint(model_dir: str, key):
    model, _ = load_embedding_model(model_dir, key)
    ckpts = sorted(glob.glob(os.path.join(model_dir, 'model_*.eqx')))
    if not ckpts:
        return model, os.path.join(model_dir, 'model.eqx')
    with open(ckpts[-1], 'rb') as f:
        model = eqx.tree_deserialise_leaves(f, model)
    return model, ckpts[-1]


@eqx.filter_jit
def belief_panel_metrics(model, parts_b, weights_b, norm_mean, norm_std, num_samples, keys):
    def _one(parts, w, key):
        gen_key, ref_key = jax.random.split(key)
        features = (parts - norm_mean) / norm_std
        embedding = model.embed(features, w)[0]
        gen = model.generate(embedding, num_samples, key=gen_key)
        gen = gen * norm_std + norm_mean
        ref_idx = jax.random.choice(ref_key, w.shape[0], (num_samples,), p=w)
        logp_out = model.log_prob(features, embedding)
        logps = logp_out[0] if isinstance(logp_out, tuple) else logp_out
        plugin_loss = -jnp.sum(w * logps)
        return gen, parts[ref_idx], plugin_loss

    return jax.vmap(_one)(parts_b, weights_b, keys)


def main() -> None:
    parser = argparse.ArgumentParser(description='L96 training monitor (MMD panel vs reference).')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_beliefs', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    out_dir = args.output_dir or os.path.join(args.model_dir, 'monitor')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.model_dir, 'data_config.json')) as f:
        data_cfg = json.load(f)
    K = data_cfg['K']
    rows = data_cfg['episode_length'] + 1
    num_scenarios = data_cfg['num_scenarios']
    num_val = data_cfg['num_val_scenarios']
    norm_mean, norm_std = data_cfg['norm_mean'], data_cfg['norm_std']

    dpath = os.path.join(args.data_path, data_dirname(K))
    data = load_arrays(dpath, ('particles_tp1', 'weights_tp1', 'meta'))
    parts_mm, weights_mm, meta_mm = data['particles_tp1'], data['weights_tp1'], data['meta']

    val_start = (num_scenarios - num_val) * rows
    val_stop = num_scenarios * rows
    idx = np.linspace(val_start, val_stop - 1, args.num_beliefs).astype(int)
    idx = idx[meta_mm[idx, 2] > 0]

    parts_b = jnp.asarray(np.asarray(parts_mm[idx], dtype=np.float32))
    weights_b = np.asarray(weights_mm[idx], dtype=np.float64)
    weights_b = jnp.asarray(weights_b / weights_b.sum(axis=1, keepdims=True), dtype=jnp.float32)
    forcing = np.asarray(meta_mm[idx, 0])
    step = np.asarray(meta_mm[idx, 2])

    model, ckpt_path = load_latest_checkpoint(args.model_dir, jax.random.PRNGKey(0))
    print(f'Loaded {ckpt_path}; {len(idx)} validation beliefs')

    keys = jax.random.split(jax.random.PRNGKey(args.seed), len(idx))
    gen_b, ref_b, plugin_b = belief_panel_metrics(
        model, parts_b, weights_b, norm_mean, norm_std, args.num_samples, keys
    )
    mmd = np.asarray(mmd2_multibw_per_step_batch(gen_b[:, None], ref_b[:, None]))[:, 0]
    plugin = np.asarray(plugin_b)

    ckpt_name = os.path.basename(ckpt_path)
    print(
        f'{ckpt_name}: mmd2 mean={mmd.mean():.5f} median={np.median(mmd):.5f} '
        f'max={mmd.max():.5f}; plugin loss mean={plugin.mean():.4f}'
    )

    csv_path = os.path.join(out_dir, 'monitor.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                ['time', 'checkpoint', 'mmd2_mean', 'mmd2_median', 'mmd2_max', 'plugin_loss_mean']
            )
        w.writerow(
            [
                time.strftime('%Y-%m-%dT%H:%M:%S'),
                ckpt_name,
                f'{mmd.mean():.6f}',
                f'{np.median(mmd):.6f}',
                f'{mmd.max():.6f}',
                f'{plugin.mean():.6f}',
            ]
        )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sc = axes[0].scatter(forcing, mmd, c=step, cmap='viridis', s=18)
    axes[0].set_xlabel('forcing F')
    axes[0].set_ylabel('per-belief MMD$^2$')
    fig.colorbar(sc, ax=axes[0], label='step')
    axes[1].scatter(step, plugin, c=forcing, cmap='coolwarm', s=18)
    axes[1].set_xlabel('step')
    axes[1].set_ylabel('plug-in loss')
    hist = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding=None)
    hist_mean = np.atleast_1d(hist['mmd2_mean'].astype(float))
    axes[2].plot(hist_mean, marker='o')
    axes[2].set_xlabel('monitor call')
    axes[2].set_ylabel('MMD$^2$ mean')
    fig.suptitle(f'{ckpt_name} — {len(idx)} validation beliefs')
    fig.tight_layout()
    pdf_path = os.path.join(out_dir, 'monitor_panel.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f'Appended {csv_path}; wrote {pdf_path}')


if __name__ == '__main__':
    main()
