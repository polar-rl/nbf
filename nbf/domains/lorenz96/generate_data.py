import argparse
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from tqdm import tqdm

from nbf.data import allocate_memmaps, load_arrays
from nbf.domains.lorenz96.config import (
    Config,
    DataConfig,
    load_config,
    resolve_runtime_paths,
)
from nbf.domains.lorenz96.env import (
    Lorenz96Env,
    build_lorenz96_env,
    conditional_gaussian_redraw,
    rk4_window,
)
from nbf.filters.particle import systematic_indices

MAX_ABS_FP16 = 60000.0


def data_dirname(K: int) -> str:
    return f'l96_belief_data_K{K}'


@eqx.filter_jit
def scenario_block(
    env: Lorenz96Env,
    num_steps: int,
    num_pf_particles: int,
    num_samples: int,
    block_keys: Array,
):
    obs_idx = env.obs_idx

    def one(key):
        gen_key, prior_key, row0_key, scan_key = jax.random.split(key, 4)
        state0 = env.generator(gen_key)
        forcing, sigma_obs = state0.forcing, state0.sigma_obs
        particles = state0.center + env.sigma_init * jax.random.normal(
            prior_key, (num_pf_particles, env.K)
        )
        w0 = jnp.full(num_pf_particles, 1.0 / num_pf_particles)

        def _subsample(parts, weights, key):
            idx = jax.random.choice(key, num_pf_particles, (num_samples,), replace=False)
            w = weights[idx]
            return parts[idx], w / (jnp.sum(w) + 1e-30)

        p0, sw0 = _subsample(particles, w0, row0_key)

        def step(carry, k):
            x_true, parts = carry
            true_key, obs_key, prop_key, sub_key, res_key, rej_key = jax.random.split(k, 6)

            x_true = rk4_window(x_true, forcing, env.dt, env.substeps, env.linear)
            x_true = x_true + env.sigma_dyn * jax.random.normal(true_key, x_true.shape)
            y = x_true[obs_idx] + sigma_obs * jax.random.normal(obs_key, (env.num_obs,))

            means = rk4_window(parts, forcing, env.dt, env.substeps, env.linear)
            marg_var = env.sigma_dyn**2 + sigma_obs**2
            log_w = -0.5 * jnp.sum((means[:, obs_idx] - y) ** 2, axis=-1) / marg_var
            w = jax.nn.softmax(log_w)

            sub_idx = jax.random.choice(sub_key, num_pf_particles, (num_samples,), replace=False)
            ps = conditional_gaussian_redraw(
                means[sub_idx], y, obs_idx, env.sigma_dyn, sigma_obs, prop_key
            )
            ws = w[sub_idx] / (jnp.sum(w[sub_idx]) + 1e-30)

            idx = systematic_indices(w, res_key)
            parts = conditional_gaussian_redraw(
                means[idx], y, obs_idx, env.sigma_dyn, sigma_obs, rej_key
            )
            return (x_true, parts), (ps, ws, x_true, y)

        _, (ps, ws, truth, obs) = jax.lax.scan(
            step, (state0.x, particles), jax.random.split(scan_key, num_steps)
        )
        ptp1 = jnp.concatenate([p0[None], ps], axis=0)
        wtp1 = jnp.concatenate([sw0[None], ws], axis=0)
        return ptp1, wtp1, truth, obs, forcing, sigma_obs, state0.center

    return jax.vmap(one)(block_keys)


def generate_data(
    data_cfg: DataConfig,
    num_samples: int,
    *,
    key: Array,
    out_dir: str,
) -> None:
    env = build_lorenz96_env(data_cfg)
    B = data_cfg.block_size
    S = data_cfg.num_scenarios
    T = data_cfg.episode_length
    if S % B != 0:
        raise ValueError(f'num_scenarios ({S}) must be divisible by block_size ({B})')
    rows = T + 1
    mm = allocate_memmaps(
        out_dir,
        {
            'particles_tp1': (np.float16, (S * rows, num_samples, data_cfg.K)),
            'weights_tp1': (np.float32, (S * rows, num_samples)),
            'meta': (np.float32, (S * rows, 3)),
        },
    )

    truth_sum = 0.0
    truth_sumsq = 0.0
    truth_count = 0
    step_col = np.tile(np.arange(rows, dtype=np.float32), B)

    for b in tqdm(range(S // B), desc='Scenario blocks (gold SMC)'):
        key, block_key = jax.random.split(key)
        ptp1, wtp1, truth, _, forcing, sigma_obs, _ = scenario_block(
            env, T, data_cfg.num_pf_particles, num_samples, jax.random.split(block_key, B)
        )
        ptp1 = np.asarray(ptp1)
        max_abs = float(np.max(np.abs(ptp1)))
        if max_abs >= MAX_ABS_FP16:
            raise ValueError(
                f'fp16 overflow guard tripped (max |x| = {max_abs:.1f}): diverged RK4?'
            )

        start = b * B * rows
        stop = start + B * rows
        mm['particles_tp1'][start:stop] = ptp1.reshape(B * rows, num_samples, data_cfg.K)
        mm['weights_tp1'][start:stop] = np.asarray(wtp1).reshape(B * rows, num_samples)
        mm['meta'][start:stop] = np.stack(
            [
                np.repeat(np.asarray(forcing, dtype=np.float32), rows),
                np.repeat(np.asarray(sigma_obs, dtype=np.float32), rows),
                step_col,
            ],
            axis=1,
        )

        truth = np.asarray(truth, dtype=np.float64)
        truth_sum += truth.sum()
        truth_sumsq += (truth**2).sum()
        truth_count += truth.size

    for m in mm.values():
        m.flush()

    mean = truth_sum / truth_count
    std = float(np.sqrt(truth_sumsq / truth_count - mean**2))
    with open(os.path.join(out_dir, 'norm_stats.json'), 'w') as f:
        json.dump({'norm_mean': float(mean), 'norm_std': std}, f, indent=2)


def load_data(data_path: str, K: int) -> dict[str, np.ndarray]:
    dpath = os.path.join(data_path, data_dirname(K))
    return load_arrays(dpath, ('particles_tp1', 'weights_tp1', 'meta'))


def load_norm_stats(data_path: str, K: int) -> dict[str, float]:
    with open(os.path.join(data_path, data_dirname(K), 'norm_stats.json')) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate Lorenz-96 belief data (gold SMC).')
    parser.add_argument('--config', type=str, default=None, help='YAML config path')
    parser.add_argument(
        '--set', nargs='*', default=[], help='Override config values: key.path=value'
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config is not None else Config()

    if args.set:
        from nbf.config import apply_overrides

        cfg = apply_overrides(cfg, args.set)
        cfg = resolve_runtime_paths(cfg)

    os.makedirs(cfg.data_path, exist_ok=True)
    out_path = os.path.join(cfg.data_path, data_dirname(cfg.data.K))

    if os.path.exists(out_path):
        print(f'Data already exists at {out_path} — skipping generation.')
        return

    print(
        f'Generating belief data via gold SMC '
        f'(K={cfg.data.K}, N={cfg.data.num_pf_particles}, '
        f'{cfg.data.num_scenarios} scenarios) ...'
    )
    key = jax.random.PRNGKey(cfg.seed)
    key, data_key = jax.random.split(key)
    generate_data(
        cfg.data,
        num_samples=cfg.training.num_samples,
        key=data_key,
        out_dir=out_path,
    )
    print(f'Saved corpus to {out_path}/')


if __name__ == '__main__':
    main()
