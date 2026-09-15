"""Mechanism ablation (rebuttal E3), reference-free.

Within one filter step, does fit-and-regenerate add or lose information relative to
the raw weighted set the encoder was given? The NBF filter is rolled on-policy along
held-out-F episodes at a pinned observation sigma; at every step the weighted set S_t
the filter would embed is scored two ways against the true state x_t:

  raw    an equal-weight resample of S_t (what a particle filter would report)
  regen  `num_ref_samples` draws from the generator conditioned on embed(S_t)

with RMSE, CRPS and (for regen) the NLL of x_t, across a sigma sweep that runs past
the training range. The within-step effect is regen minus raw at each sigma; the
cross-step effect (impoverishment compounding) is what the sigma cell of run_grid
measures by running whole filters.
"""

import argparse
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nbf.config import build_dataclass
from nbf.domains.lorenz96.config import DataConfig
from nbf.domains.lorenz96.run_grid import HELD_OUT_F, SIGMA_SWEEP, make_cell_env
from nbf.domains.lorenz96.train import simulate_episodes
from nbf.eval._tex import fmt_val
from nbf.eval.base import load_embedding_model, parse_seeds
from nbf.eval.dispatch import NUM_REF_SAMPLES
from nbf.eval.grid import atomic_savez
from nbf.eval.metrics import aggregate_mean_sem, crps_per_step_batch, rmse_per_step_batch
from nbf.filters.particle import systematic_indices
from nbf.train import make_filter, rollout_particle_sets


@eqx.filter_jit
def score_sets_batch(env, model, feats, ws, x_true, num_ref_samples, key):
    """feats (B, R, n, K) normalized, ws (B, R, n), x_true (B, R, K) raw.
    Returns raw and regenerated sample sets in raw units, (B, R, S, K) each, and the
    log-density of x_true under the regenerated belief."""
    B, R = ws.shape[:2]
    keys = jax.random.split(key, B * R * 3).reshape(B, R, 3, 2)

    def _one(f, w, x, k):
        res_key, pick_key, gen_key = k[0], k[1], k[2]
        idx = systematic_indices(w, res_key)
        raw = f[idx][jax.random.randint(pick_key, (num_ref_samples,), 0, w.shape[0])]
        emb = model.embed(f, w)[0]
        gen = model.generate(emb, num_ref_samples, key=gen_key)
        logp_out = model.log_prob(((x - env.norm_mean) / env.norm_std)[None], emb)
        logp = (logp_out[0] if isinstance(logp_out, tuple) else logp_out)[0]
        denorm = lambda z: z * env.norm_std + env.norm_mean  # noqa: E731
        return denorm(raw), denorm(gen), logp + env.normalize_log_jacobian()

    return jax.vmap(jax.vmap(_one))(feats, ws, x_true, keys)


def run_cell(args, data_cfg, model, sigma, budget, seed):
    env = make_cell_env(data_cfg, HELD_OUT_F, (sigma, sigma), args.episode_length)
    key = jax.random.PRNGKey(seed)
    ep_key, reset_key, step_key, score_key = jax.random.split(key, 4)
    eps = simulate_episodes(env, args.num_episodes, ep_key)
    B, T = args.num_episodes, args.episode_length
    filt = make_filter(model, args.proposal, budget)
    feats, ws, neffs = rollout_particle_sets(
        env, filt, eps.state0, eps.obs0, eps.states_next, eps.obs_prev, eps.obs_next,
        jax.random.split(reset_key, B), jax.random.split(step_key, B * T).reshape(B, T, -1),
    )
    # Row 0 is the prior set; score the T update rows against x_1..x_T.
    x_true = eps.states_next.x
    raw, regen, logp = score_sets_batch(
        env, model, feats[:, 1:], ws[:, 1:], x_true, args.num_ref_samples, score_key
    )
    return {
        'raw_rmse': np.asarray(rmse_per_step_batch(raw, x_true)),
        'raw_crps': np.asarray(crps_per_step_batch(raw, x_true)),
        'regen_rmse': np.asarray(rmse_per_step_batch(regen, x_true)),
        'regen_crps': np.asarray(crps_per_step_batch(regen, x_true)),
        'regen_nll': np.asarray(-logp),
        'ess_frac': np.asarray(neffs) / budget,
    }


METRICS = ('raw_rmse', 'regen_rmse', 'raw_crps', 'regen_crps', 'regen_nll', 'ess_frac')


def write_summary(out_dir, results, sigmas, budgets, t_min):
    md = ['# Mechanism ablation: raw weighted set vs fit-and-regenerate, held-out F', '',
          '| sigma | n | raw RMSE | regen RMSE | raw CRPS | regen CRPS | regen NLL | ESS/n |',
          '|---|---|---|---|---|---|---|---|']
    tex = [r'\begin{tabular}{rrcccccc}',
           r'$\sigma_{\mathrm{obs}}$ & $n$ & raw RMSE & regen RMSE & raw CRPS & regen CRPS & regen NLL & ESS/$n$ \\',
           r'\hline']
    for sigma in sigmas:
        for budget in budgets:
            pooled = {m: np.concatenate([r[m] for (s, b, _), r in results.items() if s == sigma and b == budget])
                      for m in METRICS}
            cells = []
            for m in METRICS:
                mean, sem = aggregate_mean_sem(pooled[m][:, t_min:])
                cells.append(f'{fmt_val(mean)} ± {fmt_val(sem)}' if m != 'ess_frac' else fmt_val(mean))
            md.append(f'| {sigma:g} | {budget} | ' + ' | '.join(cells) + ' |')
            tex.append(f'{sigma:g} & {budget} & ' + ' & '.join(cells) + r' \\')
    tex.append(r'\end{tabular}')
    with open(os.path.join(out_dir, 'e3_summary.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    with open(os.path.join(out_dir, 'e3.tex'), 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print(f'Saved {out_dir}/e3_summary.md and e3.tex')


def main() -> None:
    parser = argparse.ArgumentParser(description='Mechanism ablation: raw set vs regenerate.')
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sigmas', default=','.join(f'{s:g}' for s in SIGMA_SWEEP))
    parser.add_argument('--budgets', default='1024,2048')
    parser.add_argument('--proposal', default='optimal', choices=['bootstrap', 'optimal'])
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--episode_length', type=int, default=0)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--num_seeds', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_ref_samples', type=int, default=NUM_REF_SAMPLES)
    parser.add_argument('--t_min', type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, 'data_config.json')) as f:
        data_cfg = build_dataclass(DataConfig, json.load(f))
    args.episode_length = args.episode_length or data_cfg.episode_length
    model, _ = load_embedding_model(args.model_dir, jax.random.PRNGKey(0))
    sigmas = [float(s) for s in args.sigmas.split(',') if s.strip()]
    budgets = [int(b) for b in args.budgets.split(',') if b.strip()]
    results = {}
    for sigma in sigmas:
        for budget in budgets:
            for seed in parse_seeds(args):
                res = run_cell(args, data_cfg, model, sigma, budget, seed)
                results[(sigma, budget, seed)] = res
                rr, _ = aggregate_mean_sem(res['raw_crps'][:, args.t_min:])
                gr, _ = aggregate_mean_sem(res['regen_crps'][:, args.t_min:])
                print(f'  sigma={sigma:g} n={budget} seed={seed}: crps raw={rr:.3f} regen={gr:.3f}')
    atomic_savez(os.path.join(args.output_dir, 'e3.npz'),
                 **{f'{s:g}|{b}|s{seed}|{m}': v for (s, b, seed), r in results.items() for m, v in r.items()})
    write_summary(args.output_dir, results, sigmas, budgets, args.t_min)


if __name__ == '__main__':
    main()
