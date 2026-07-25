import argparse
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nbf.domains.lorenz96.config import DataConfig
from nbf.domains.lorenz96.env import build_lorenz96_env, conditional_gaussian_redraw, rk4_window
from nbf.domains.lorenz96.eval_utils import compute_method_samples
from nbf.eval.base import rollout_episodes, split_T
from nbf.eval.dispatch import GOLD_SMC_COUNT
from nbf.eval.metrics import mmd2_multibw_per_step_batch
from nbf.filters.particle import systematic_indices


@eqx.filter_jit
def gold_smc_diagnostic_stream_batch(
    env,
    state0_b,
    obs_values_b,
    prior_keys,
    step_keys_b,
    num_particles: int,
    num_ref_samples: int,
):
    obs_idx = env.obs_idx

    def _one(state0, obs_values, prior_key, step_keys):
        forcing, sigma_obs = state0.forcing, state0.sigma_obs
        parts = state0.center + env.sigma_init * jax.random.normal(
            prior_key, (num_particles, env.K)
        )

        def _step(parts, xs):
            k, y = xs
            prop_key, res_key, rej_key, samp_key = jax.random.split(k, 4)
            means = rk4_window(parts, forcing, env.dt, env.substeps, env.linear)
            parts = means + env.sigma_dyn * jax.random.normal(prop_key, parts.shape)
            log_w = -0.5 * jnp.sum(((parts[:, obs_idx] - y) / sigma_obs) ** 2, axis=-1)
            w = jax.nn.softmax(log_w)
            ess = 1.0 / jnp.sum(w**2)
            idx = systematic_indices(w, res_key)
            unique = jnp.sum(jnp.bincount(idx, length=num_particles) > 0) / num_particles
            parts = conditional_gaussian_redraw(
                means[idx], y, obs_idx, env.sigma_dyn, sigma_obs, rej_key
            )
            samp_idx = jax.random.randint(samp_key, (num_ref_samples,), 0, num_particles)
            return parts, (parts[samp_idx], ess, unique)

        _, out = jax.lax.scan(_step, parts, (step_keys, obs_values))
        return out

    return jax.vmap(_one)(state0_b, obs_values_b, prior_keys, step_keys_b)


def main() -> None:
    parser = argparse.ArgumentParser(description='Gold-SMC reference self-consistency check.')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--episode_length', type=int, default=20)
    parser.add_argument('--num_particles', type=int, default=GOLD_SMC_COUNT)
    parser.add_argument('--num_ref_samples', type=int, default=1024)
    parser.add_argument('--scale_method', type=str, default='pf_8192')
    parser.add_argument('--gate_frac', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--forcing',
        type=float,
        default=None,
        help='Pin F to this value (e.g. 8.0 for the held-out headline); default = training bands.',
    )
    args = parser.parse_args()

    data_cfg = (
        DataConfig(K=args.K)
        if args.forcing is None
        else DataConfig(K=args.K, f_bands=[[args.forcing, args.forcing]])
    )
    env = build_lorenz96_env(data_cfg, time_limit=args.episode_length)
    key = jax.random.PRNGKey(args.seed)
    rollout_key, run1_key, run2_key, method_key = jax.random.split(key, 4)
    state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = rollout_episodes(
        env, args.episode_length, args.num_episodes, rollout_key
    )
    N, T = args.num_episodes, args.episode_length

    def _run(k):
        pk, sk = jax.random.split(k)
        return gold_smc_diagnostic_stream_batch(
            env,
            state0_b,
            obs_next_b.values,
            jax.random.split(pk, N),
            split_T(sk, N, T),
            args.num_particles,
            args.num_ref_samples,
        )

    forcing_desc = 'training bands' if args.forcing is None else f'F={args.forcing}'
    print(
        f'Two gold-SMC runs: K={args.K}, N={args.num_particles}, '
        f'{N} episodes x T={T}, S={args.num_ref_samples}, {forcing_desc}'
    )
    samps1, ess1, uniq1 = _run(run1_key)
    samps2, ess2, uniq2 = _run(run2_key)
    inter = np.asarray(mmd2_multibw_per_step_batch(samps1, samps2))

    scale_samples = compute_method_samples(
        args.scale_method,
        env=env,
        embedding_model=None,
        state0_b=state0_b,
        obs0_b=obs0_b,
        states_next_b=states_next_b,
        obs_prev_b=obs_prev_b,
        obs_next_b=obs_next_b,
        num_ref_samples=args.num_ref_samples,
        key=method_key,
    )
    scale = np.asarray(mmd2_multibw_per_step_batch(scale_samples, samps1))

    ess_frac = np.asarray(ess1) / args.num_particles
    uniq = np.asarray(uniq1)
    print('per-step means over episodes:')
    print(f'  t   inter-run MMD2   {args.scale_method} MMD2   ESS/n   unique-frac')
    for t in range(T):
        print(
            f'  {t:<3} {inter[:, t].mean():14.6f} {scale[:, t].mean():10.6f} '
            f'{ess_frac[:, t].mean():7.4f} {uniq[:, t].mean():11.4f}'
        )
    print(
        f'run-2 diagnostics match: ESS/n={np.asarray(ess2).mean() / args.num_particles:.4f} '
        f'unique-frac={np.asarray(uniq2).mean():.4f}'
    )

    inter_mean = float(inter.mean())
    scale_mean = float(scale.mean())
    ok = inter_mean < args.gate_frac * scale_mean
    print(
        f'inter-run MMD2 mean = {inter_mean:.6f}; {args.scale_method}-vs-reference mean = '
        f'{scale_mean:.6f}; gate: inter < {args.gate_frac} x method'
    )
    print(f'reference self-consistency {"PASS" if ok else "FAIL"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
