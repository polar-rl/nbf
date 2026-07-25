import argparse

import jax
import numpy as np

from nbf.domains.lorenz96.env import Lorenz96Env
from nbf.domains.lorenz96.eval_utils import sample_l96_negatives_batch
from nbf.eval.base import rollout_episodes, split_T
from nbf.eval.metrics import density_metrics_from_logps
from nbf.eval.samples import pf_samples_density_stream_batch
from nbf.filters.particle import ParticleFilter


def main() -> None:
    parser = argparse.ArgumentParser(description='Sweep PF-KDE bandwidths for L96 (LDS tune).')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--counts', type=str, default='1024,8192,65536')
    parser.add_argument('--bandwidths', type=str, default='0.5,0.75,1.0,1.5,2.0,3.0,4.0')
    parser.add_argument('--num_episodes', type=int, default=8)
    parser.add_argument('--episode_length', type=int, default=20)
    parser.add_argument('--num_negatives', type=int, default=256)
    parser.add_argument('--norm_mean', type=float, default=2.3)
    parser.add_argument('--norm_std', type=float, default=3.6)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    env = Lorenz96Env(
        K=args.K,
        time_limit=args.episode_length,
        norm_mean=args.norm_mean,
        norm_std=args.norm_std,
    )
    key = jax.random.PRNGKey(args.seed)
    rollout_key, neg_key, method_key = jax.random.split(key, 3)
    state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = rollout_episodes(
        env, args.episode_length, args.num_episodes, rollout_key
    )
    N, T = args.num_episodes, args.episode_length
    neg_pts_b = sample_l96_negatives_batch(
        env, jax.random.split(neg_key, N * T).reshape(N, T, -1), args.num_negatives
    )

    counts = [int(c) for c in args.counts.split(',')]
    bandwidths = [float(h) for h in args.bandwidths.split(',')]
    print(f'K={args.K}: sweeping counts {counts} x bandwidths {bandwidths}')
    print('(selection: min normalized NLL; the KDE logp is unnormalized, and the')
    print(' Gaussian constant cancels in LDS, so raw NLL/LDS are monotone in h)')
    for n in counts:
        pf = ParticleFilter(num_particles=n)
        rk, sk, samp_k = jax.random.split(jax.random.fold_in(method_key, n), 3)
        rows = []
        for h in bandwidths:
            _, lp_pos, lp_neg = pf_samples_density_stream_batch(
                env,
                pf,
                state0_b,
                obs0_b,
                states_next_b,
                obs_prev_b,
                obs_next_b,
                jax.random.split(rk, N),
                split_T(sk, N, T),
                split_T(samp_k, N, T),
                neg_pts_b,
                1,
                h,
            )
            m = density_metrics_from_logps(lp_pos, lp_neg)
            lds = float(np.mean(np.asarray(m['log_density_score'])))
            nll = float(np.mean(np.asarray(m['nll'])))
            auc = float(np.mean(np.asarray(m['auc'])))
            nll_norm = nll + 0.5 * args.K * float(np.log(2.0 * np.pi * h * h))
            rows.append((h, lds, nll, nll_norm, auc))
            print(
                f'  n={n:<6} h={h:<5} lds={lds:9.3f} nll={nll:9.3f} '
                f'nll_norm={nll_norm:9.3f} auc={auc:.4f}'
            )
        best = min(rows, key=lambda r: r[3])
        print(f'  n={n}: best h={best[0]} (nll_norm={best[3]:.3f})')


if __name__ == '__main__':
    main()
