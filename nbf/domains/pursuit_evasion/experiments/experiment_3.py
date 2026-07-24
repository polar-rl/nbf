import os

import jax
import matplotlib.pyplot as plt
import numpy as np

from nbf.domains.pursuit_evasion.eval.base import make_env_kn
from nbf.domains.pursuit_evasion.eval.dispatch import (
    compute_method_samples_and_density,
    prepare_oracle_and_negatives,
)
from nbf.domains.pursuit_evasion.policies import (
    sample_ood_component_policy,
    sample_train_mixture_policy,
)
from nbf.eval._tex import latex_escape
from nbf.eval.base import parse_csv_methods, rollout_episodes
from nbf.eval.dispatch import NUM_NEGATIVES, NUM_REF_SAMPLES
from nbf.eval.metrics import mmd2_per_step_batch

DEFAULT_METHODS = ('nbf_1024', 'pf_8192')
REGIMES = ('in_dist', 'ood_wall', 'ood_orbit', 'ood_sharp_goal', 'ood_diffuse_goal')


def _sample_policy(regime: str, key):
    if regime == 'in_dist':
        return sample_train_mixture_policy(key)
    if regime == 'ood_wall':
        return sample_ood_component_policy(key, 'wall')
    if regime == 'ood_orbit':
        return sample_ood_component_policy(key, 'orbit')
    if regime == 'ood_sharp_goal':
        return sample_ood_component_policy(key, 'sharp_goal')
    if regime == 'ood_diffuse_goal':
        return sample_ood_component_policy(key, 'diffuse_goal')
    raise ValueError(f'unknown regime: {regime}')


def run(args, embedding_model, key):
    if args.num_evaders != 1:
        raise ValueError(f'E3 is k=1 only, got num_evaders={args.num_evaders}')
    methods = parse_csv_methods(args, 'e3_methods', DEFAULT_METHODS)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    regime_keys = jax.random.split(key, len(REGIMES))
    per_regime: dict[str, dict[str, dict]] = {}
    for rk, regime in zip(regime_keys, REGIMES, strict=True):
        print('=== regime:', regime)
        per_regime[regime] = _run_one_regime(args, embedding_model, methods, regime, rk)

    npz_path = os.path.join(out_dir, 'e3_per_regime.npz')
    np.savez(
        npz_path,
        regimes=np.asarray(REGIMES),
        **{
            f'{regime}/{m}/mmd_per_episode': np.asarray(per_regime[regime][m]['mmd_per_episode'])
            for regime in REGIMES
            for m in methods
        },
        **{
            f'{regime}/{m}/lds_per_episode': np.asarray(per_regime[regime][m]['lds_per_episode'])
            for regime in REGIMES
            for m in methods
        },
    )
    print(f'  wrote {npz_path}')

    tex_path = os.path.join(out_dir, 'e3_table.tex')
    with open(tex_path, 'w') as fh:
        fh.write(_make_tex_table(per_regime, methods))
    print(f'  wrote {tex_path}')

    _plot_e3(per_regime, methods, out_dir)


def _run_one_regime(args, embedding_model, methods, regime, key):
    N = args.num_episodes
    T = args.episode_length
    k = 1

    pol_keys = jax.random.split(jax.random.fold_in(key, 1), N)
    rollout_keys = jax.random.split(jax.random.fold_in(key, 2), N)
    oracle_keys = jax.random.split(jax.random.fold_in(key, 3), N)
    method_keys = jax.random.split(jax.random.fold_in(key, 4), N)
    neg_keys = jax.random.split(jax.random.fold_in(key, 5), N)

    mmd_per_ep = {m: np.zeros((N, T), dtype=np.float64) for m in methods}
    lds_per_ep = {m: np.full((N,), np.nan, dtype=np.float64) for m in methods}

    for ep in range(N):
        policy = _sample_policy(regime, pol_keys[ep])
        env = make_env_kn(k, T, (policy,))

        s0_b, o0_b, sn_b, op_b, on_b = rollout_episodes(env, T, 1, rollout_keys[ep])

        oracle, neg_pts_b = prepare_oracle_and_negatives(
            out_dir=args.output_dir,
            env=env,
            k=k,
            scenario=f'{regime}_ep{ep}',
            num_episodes=1,
            episode_length=T,
            seed=ep,
            num_ref_samples=NUM_REF_SAMPLES,
            num_negatives=NUM_NEGATIVES,
            state0_b=s0_b,
            obs0_b=o0_b,
            states_next_b=sn_b,
            obs_prev_b=op_b,
            obs_next_b=on_b,
            oracle_key=oracle_keys[ep],
            neg_key=neg_keys[ep],
        )

        ep_method_keys = jax.random.split(method_keys[ep], len(methods))
        for i, spec in enumerate(methods):
            samples, dm = compute_method_samples_and_density(
                spec,
                env=env,
                embedding_model=embedding_model,
                k=k,
                state0_b=s0_b,
                obs0_b=o0_b,
                states_next_b=sn_b,
                obs_prev_b=op_b,
                obs_next_b=on_b,
                neg_pts_b=neg_pts_b,
                num_ref_samples=NUM_REF_SAMPLES,
                key=ep_method_keys[i],
            )
            mmd_ps = np.asarray(mmd2_per_step_batch(samples, oracle))[0]
            mmd_per_ep[spec][ep] = mmd_ps
            if dm is not None:
                lds_per_ep[spec][ep] = float(np.asarray(dm['log_density_score']).mean())

        if (ep + 1) % max(N // 10, 1) == 0 or ep == N - 1:
            tag = ', '.join(f'{m}={mmd_per_ep[m][: ep + 1].mean():.4f}' for m in methods)
            print(f'  {regime} [{ep + 1}/{N}] {tag}')

    out = {}
    for m in methods:
        mmd_per_ep_mean = mmd_per_ep[m].mean(axis=1)
        out[m] = {
            'mmd_per_episode': mmd_per_ep_mean,
            'mmd_mean': float(mmd_per_ep_mean.mean()),
            'mmd_se': float(mmd_per_ep_mean.std(ddof=1) / np.sqrt(N)) if N > 1 else 0.0,
            'lds_per_episode': lds_per_ep[m],
            'lds_mean': float(np.nanmean(lds_per_ep[m]))
            if np.isfinite(lds_per_ep[m]).any()
            else float('nan'),
            'lds_se': (
                float(np.nanstd(lds_per_ep[m], ddof=1) / np.sqrt(np.isfinite(lds_per_ep[m]).sum()))
                if np.isfinite(lds_per_ep[m]).sum() > 1
                else 0.0
            ),
        }
        print(
            f'  [{regime}] {m}: MMD^2={out[m]["mmd_mean"]:.4f}±{out[m]["mmd_se"]:.4f} '
            f'LDS={out[m]["lds_mean"]:.4f}±{out[m]["lds_se"]:.4f}'
        )
    return out


def _make_tex_table(per_regime, methods):
    cols = 'l' + 'rr' * len(methods)
    header = ' & '.join(
        ['regime'] + [f'{m} MMD$^2$' for m in methods] + [f'{m} LDS' for m in methods]
    )
    cols = 'l' + 'r' * (2 * len(methods))
    lines = [
        f'\\begin{{tabular}}{{{cols}}}',
        '\\toprule',
        header + ' \\\\',
        '\\midrule',
    ]
    for regime in REGIMES:
        row = [latex_escape(regime)]
        for m in methods:
            s = per_regime[regime][m]
            row.append(f'{s["mmd_mean"]:.4f} $\\pm$ {s["mmd_se"]:.4f}')
        for m in methods:
            s = per_regime[regime][m]
            row.append(f'{s["lds_mean"]:.3f} $\\pm$ {s["lds_se"]:.3f}')
        lines.append(' & '.join(row) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


def _plot_e3(per_regime, methods, out_dir):
    from nbf.domains.pursuit_evasion.visualize import series_style

    xs = np.arange(len(REGIMES))
    n_methods = len(methods)
    width = 0.8 / n_methods
    labels = [r.replace('ood_', '').replace('_', ' ') for r in REGIMES]

    specs = [
        ('mmd', r'MMD$^2$', 'e3_mmd.pdf'),
        ('lds', 'log-density score', 'e3_lds.pdf'),
    ]
    for metric, ylabel, fname in specs:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for i, m in enumerate(methods):
            offset = (i - (n_methods - 1) / 2) * width
            means = [per_regime[r][m][f'{metric}_mean'] for r in REGIMES]
            ses = [per_regime[r][m][f'{metric}_se'] for r in REGIMES]
            label, color = series_style(m)
            ax.bar(xs + offset, means, width, yerr=ses, color=color, label=label, capsize=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
        ax.axvline(0.5, color='gray', linestyle=':', linewidth=0.8)
        ax.legend(fontsize=10)
        fig.tight_layout()
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f'  wrote {out_path}')


def replot(args):
    out_dir = args.output_dir
    npz_path = os.path.join(out_dir, 'e3_per_regime.npz')
    d = np.load(npz_path, allow_pickle=False)
    methods_seen: list[str] = []
    per_regime: dict[str, dict[str, dict]] = {r: {} for r in REGIMES}
    for key in d.files:
        if '/' not in key:
            continue
        parts = key.split('/')
        if len(parts) != 3:
            continue
        regime, m, field = parts
        if regime not in per_regime:
            continue
        per_regime[regime].setdefault(m, {})
        arr = np.asarray(d[key])
        if field == 'mmd_per_episode':
            per_regime[regime][m]['mmd_per_episode'] = arr
            per_regime[regime][m]['mmd_mean'] = float(arr.mean())
            per_regime[regime][m]['mmd_se'] = (
                float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            )
        elif field == 'lds_per_episode':
            per_regime[regime][m]['lds_per_episode'] = arr
            finite = np.isfinite(arr)
            per_regime[regime][m]['lds_mean'] = (
                float(np.nanmean(arr)) if finite.any() else float('nan')
            )
            per_regime[regime][m]['lds_se'] = (
                float(np.nanstd(arr, ddof=1) / np.sqrt(finite.sum())) if finite.sum() > 1 else 0.0
            )
        if m not in methods_seen:
            methods_seen.append(m)
    methods = parse_csv_methods(args, 'e3_methods', tuple(methods_seen))

    tex_path = os.path.join(out_dir, 'e3_table.tex')
    with open(tex_path, 'w') as fh:
        fh.write(_make_tex_table(per_regime, methods))
    print(f'  wrote {tex_path}')
    _plot_e3(per_regime, methods, out_dir)
