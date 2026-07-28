import argparse
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nbf.config import build_dataclass
from nbf.domains.lorenz96.config import DataConfig
from nbf.domains.lorenz96.env import conditional_gaussian_redraw, rk4_window
from nbf.domains.lorenz96.run_grid import make_cell_env
from nbf.eval._tex import fmt_val
from nbf.eval.base import (
    chunked_stream,
    load_embedding_model,
    parse_seeds,
    rollout_episodes,
    split_T,
)
from nbf.eval.dispatch import GOLD_SMC_COUNT, NUM_REF_SAMPLES
from nbf.eval.metrics import aggregate_mean_sem, mmd2_multibw_per_step_chunked
from nbf.filters.particle import systematic_indices

DEFAULT_SIGMAS = '0.25,0.5,1.0,1.5,2.0,3.0'
DEFAULT_BUDGETS = '1024'


@eqx.filter_jit
def e3_stream_batch(
    env,
    model,
    state0_b,
    obs_values_b,
    prior_keys,
    step_keys_b,
    num_gold,
    budget,
    num_ref_samples,
):
    obs_idx = env.obs_idx

    def _one(state0, obs_values, prior_key, step_keys):
        forcing, sigma_obs = state0.forcing, state0.sigma_obs
        parts = state0.center + env.sigma_init * jax.random.normal(prior_key, (num_gold, env.K))

        def _step(parts, xs):
            k, y = xs
            keys = jax.random.split(k, 8)
            sub_key, bprop_key, a_key, gen_key, prop_key, res_key, rej_key, samp_key = keys

            sub_idx = jax.random.choice(sub_key, num_gold, (budget,), replace=False)
            b_means = rk4_window(parts[sub_idx], forcing, env.dt, env.substeps, env.linear)
            b_parts = b_means + env.sigma_dyn * jax.random.normal(bprop_key, b_means.shape)
            b_logw = -0.5 * jnp.sum(((b_parts[:, obs_idx] - y) / sigma_obs) ** 2, axis=-1)
            b_w = jax.nn.softmax(b_logw)
            ess_frac = 1.0 / (jnp.sum(b_w**2) * budget)

            a_idx = jax.random.choice(a_key, budget, (num_ref_samples,), p=b_w)
            weighted_samps = b_parts[a_idx]

            emb = model.embed(env.normalize_flat(b_parts), b_w)[0]
            gen = model.generate(emb, num_ref_samples, key=gen_key)
            regen_samps = gen * env.norm_std + env.norm_mean

            means = rk4_window(parts, forcing, env.dt, env.substeps, env.linear)
            parts = means + env.sigma_dyn * jax.random.normal(prop_key, parts.shape)
            log_w = -0.5 * jnp.sum(((parts[:, obs_idx] - y) / sigma_obs) ** 2, axis=-1)
            w = jax.nn.softmax(log_w)
            idx = systematic_indices(w, res_key)
            parts = conditional_gaussian_redraw(
                means[idx], y, obs_idx, env.sigma_dyn, sigma_obs, rej_key
            )
            samp_idx = jax.random.randint(samp_key, (num_ref_samples,), 0, num_gold)
            return parts, (weighted_samps, regen_samps, parts[samp_idx], ess_frac)

        _, (a, b, ref, ess) = jax.lax.scan(_step, parts, (step_keys, obs_values))
        return a, b, ref, ess

    return jax.vmap(_one)(state0_b, obs_values_b, prior_keys, step_keys_b)


def run_cell(args, data_cfg, model, sigma, budget, seed):
    env = make_cell_env(
        data_cfg, ((args.f_value, args.f_value),), (sigma, sigma), args.episode_length
    )
    key = jax.random.PRNGKey(seed)
    rollout_key, prior_key, step_key = jax.random.split(key, 3)
    state0_b, _, _, _, obs_next_b = rollout_episodes(
        env, args.episode_length, args.num_episodes, rollout_key
    )
    N, T = args.num_episodes, args.episode_length
    a, b, ref, ess = chunked_stream(
        lambda *xs: e3_stream_batch(env, model, *xs),
        (state0_b, obs_next_b.values),
        (jax.random.split(prior_key, N), split_T(step_key, N, T)),
        (args.num_gold, budget, args.num_ref_samples),
        N,
        args.chunk_size,
        desc=f'e3 s{sigma} n{budget} seed{seed}',
    )
    return {
        'mmd_weighted': np.asarray(
            mmd2_multibw_per_step_chunked(a, ref, chunk_size=args.chunk_size)
        ),
        'mmd_regen': np.asarray(mmd2_multibw_per_step_chunked(b, ref, chunk_size=args.chunk_size)),
        'ess_frac': np.asarray(ess),
    }


def write_summaries(args, sigmas, budgets, results):
    md_lines = [
        f'# L96 E3 mechanism ablation (F={args.f_value}, t>={args.t_min})',
        '',
        '| sigma_obs | n | weighted set MMD$^2$ | fit-and-regenerate MMD$^2$ | ESS/n |',
        '|---|---|---|---|---|',
    ]
    tex_lines = [
        f'% e3 F={args.f_value} t_min={args.t_min}',
        r'\begin{tabular}{ccccc}',
        r'$\sigma_{obs}$ & $n$ & weighted set MMD$^2$ & fit-and-regenerate MMD$^2$ & ESS/$n$ \\',
        r'\hline',
    ]
    for sigma in sigmas:
        for budget in budgets:
            pooled = {
                metric: np.concatenate(
                    [results[(sigma, budget, s)][metric] for s in parse_seeds(args)], axis=0
                )[:, args.t_min :]
                for metric in ('mmd_weighted', 'mmd_regen', 'ess_frac')
            }
            mw, mw_sem = aggregate_mean_sem(pooled['mmd_weighted'])
            mg, mg_sem = aggregate_mean_sem(pooled['mmd_regen'])
            ess, _ = aggregate_mean_sem(pooled['ess_frac'])
            cells = [
                f'{sigma}',
                f'{budget}',
                f'{fmt_val(mw)} ± {fmt_val(mw_sem)}',
                f'{fmt_val(mg)} ± {fmt_val(mg_sem)}',
                fmt_val(ess),
            ]
            md_lines.append('| ' + ' | '.join(cells) + ' |')
            tex_lines.append(' & '.join(cells) + r' \\')
    tex_lines += [r'\end{tabular}', '']

    md_path = os.path.join(args.output_dir, 'e3_summary.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')
    tex_path = os.path.join(args.output_dir, 'e3.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines))
    print(f'Saved {md_path} and {tex_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='L96 E3 within-step mechanism ablation.')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--sigmas', type=str, default=DEFAULT_SIGMAS)
    parser.add_argument('--budgets', type=str, default=DEFAULT_BUDGETS)
    parser.add_argument('--f_value', type=float, default=8.0)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--episode_length', type=int, default=0)
    parser.add_argument('--seeds', type=str, default='0,1,2')
    parser.add_argument('--num_seeds', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--t_min', type=int, default=5)
    parser.add_argument('--num_gold', type=int, default=GOLD_SMC_COUNT)
    parser.add_argument('--num_ref_samples', type=int, default=NUM_REF_SAMPLES)
    parser.add_argument('--chunk_size', type=int, default=4)
    parser.add_argument(
        '--fm_num_steps',
        type=int,
        default=0,
        help='override flow ODE steps for generation (0 = keep model config)',
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.model_dir, 'data_config.json')) as f:
        data_cfg = build_dataclass(DataConfig, json.load(f))
    if data_cfg.norm_mean is None or data_cfg.norm_std is None:
        raise ValueError('model_dir data_config.json lacks norm stats; retrain or fix the file')
    args.episode_length = args.episode_length or data_cfg.episode_length

    model, _ = load_embedding_model(
        args.model_dir, jax.random.PRNGKey(0), fm_num_steps=args.fm_num_steps or None
    )
    print(
        f'Loaded embedding model from {args.model_dir} '
        f'(fm_num_steps={args.fm_num_steps or "config"})'
    )

    sigmas = [float(s) for s in args.sigmas.split(',') if s.strip()]
    budgets = [int(b) for b in args.budgets.split(',') if b.strip()]
    seeds = parse_seeds(args)
    print(
        f'E3: F={args.f_value}, sigmas={sigmas}, budgets={budgets}, seeds={seeds}, '
        f'{args.num_episodes} episodes x T={args.episode_length}, gold N={args.num_gold}'
    )

    results = {}
    for sigma in sigmas:
        for budget in budgets:
            for seed in seeds:
                res = run_cell(args, data_cfg, model, sigma, budget, seed)
                results[(sigma, budget, seed)] = res
                mw, mw_sem = aggregate_mean_sem(res['mmd_weighted'][:, args.t_min :])
                mg, mg_sem = aggregate_mean_sem(res['mmd_regen'][:, args.t_min :])
                ess, _ = aggregate_mean_sem(res['ess_frac'][:, args.t_min :])
                print(
                    f'  [s{sigma} n{budget} seed {seed}] '
                    f'weighted={mw:.5f}+/-{mw_sem:.5f} regen={mg:.5f}+/-{mg_sem:.5f} '
                    f'ess/n={ess:.3f}'
                )

    out = {
        f's{sigma}|n{budget}|seed{seed}|{metric}': arr
        for (sigma, budget, seed), metrics in results.items()
        for metric, arr in metrics.items()
    }
    npz_path = os.path.join(args.output_dir, 'e3.npz')
    np.savez(npz_path, **out)
    print(f'Saved {npz_path}')
    write_summaries(args, sigmas, budgets, results)


if __name__ == '__main__':
    main()
