import os

import jax
import matplotlib.pyplot as plt
import numpy as np

from nbf.domains.pursuit_evasion.eval.base import (
    load_model_for_k,
    make_env_kn,
    parse_csv_ks,
    parse_csv_methods,
    parse_seeds,
    rollout_episodes,
)
from nbf.domains.pursuit_evasion.eval.dispatch import (
    NUM_NEGATIVES,
    NUM_REF_SAMPLES,
    compute_method_samples_and_density,
    prepare_oracle_and_negatives,
)
from nbf.domains.pursuit_evasion.eval.metrics import (
    METRIC_FNS,
    aggregate_mean_sem,
    aggregate_per_step_mean_sem,
)
from nbf.domains.pursuit_evasion.experiments._tex import fmt_val
from nbf.domains.pursuit_evasion.policies import (
    TRAIN_MIXTURE_COMPONENTS,
    sample_train_component_policy,
)

DEFAULT_KS = (1, 2, 3, 5)
DEFAULT_METHODS = (
    'nbf_128',
    'nbf_512',
    'nbf_1024',
    'nbf_2048',
    'nbf_8192',
    'pf_512',
    'pf_2048',
    'pf_8192',
    'gmf_64',
)
SUMMARY_METRICS = ('mmd_multibw', 'lds')
PLOT_METRICS = (
    ('mmd_multibw', r'MMD$^2$ (multi-bw)'),
    ('lds', 'log-density score'),
)


def run(args, embedding_model, key):
    seeds = getattr(args, 'seeds_list', None) or parse_seeds(args)
    ks = parse_csv_ks(args, 'e2_ks', DEFAULT_KS)
    methods = tuple(
        m for m in parse_csv_methods(args, 'e2_methods', DEFAULT_METHODS) if m != 'pf_oracle'
    )
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    components = TRAIN_MIXTURE_COMPONENTS
    N_per_policy = max(1, args.num_episodes // len(components))
    print(
        f'E2: ks={ks}, seeds={seeds}, components={components}, '
        f'N_per_policy_per_seed={N_per_policy}, T={args.episode_length}'
    )

    per_k_data = {}
    for k in ks:
        em = embedding_model if k == args.num_evaders else load_model_for_k(args, k)
        if em is None:
            print(f'E2: no model for k={k}, skipping')
            continue
        rk = jax.random.fold_in(key, k)
        per_k_data[k] = _run_one_k(
            args, em, k, methods, seeds, components, N_per_policy, rk, out_dir
        )

    if not per_k_data:
        return

    _save_per_step_npz(out_dir, per_k_data, methods, seeds, components, ks=tuple(per_k_data))
    _write_summary(out_dir, per_k_data, methods, seeds, components)
    _plot_per_policy_grid(out_dir, per_k_data, methods, components)


def _run_one_k(args, embedding_model, k, methods, seeds, components, N_per_policy, key, out_dir):
    T = args.episode_length
    N_total = len(seeds) * len(components) * N_per_policy
    metrics = {
        name: {m: np.full((N_total, T), np.nan, dtype=np.float64) for m in methods}
        for name in ('mmd', 'mmd_multibw', 'lds')
    }
    policy_idx = np.zeros((N_total,), dtype=np.int32)
    seed_idx = np.zeros((N_total,), dtype=np.int32)

    pol_master = jax.random.fold_in(key, 1001)
    rollout_master = jax.random.fold_in(key, 1002)
    oracle_master = jax.random.fold_in(key, 1003)
    method_master = jax.random.fold_in(key, 1004)
    neg_master = jax.random.fold_in(key, 1005)

    for si, seed in enumerate(seeds):
        seed_offset = si * len(components) * N_per_policy
        for ci, comp in enumerate(components):
            base = seed_offset + ci * N_per_policy
            policy_idx[base : base + N_per_policy] = ci
            seed_idx[base : base + N_per_policy] = si

            pol_key = jax.random.fold_in(jax.random.fold_in(pol_master, seed), ci)
            rk = jax.random.fold_in(jax.random.fold_in(rollout_master, seed), ci)
            ok = jax.random.fold_in(jax.random.fold_in(oracle_master, seed), ci)
            mk_root = jax.random.fold_in(jax.random.fold_in(method_master, seed), ci)
            nk = jax.random.fold_in(jax.random.fold_in(neg_master, seed), ci)

            policy = sample_train_component_policy(pol_key, comp)
            env = make_env_kn(k, T, (policy,) * k)
            state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = rollout_episodes(
                env, T, N_per_policy, rk
            )

            oracle, neg_pts_b = prepare_oracle_and_negatives(
                out_dir=out_dir,
                env=env,
                k=k,
                scenario=comp,
                num_episodes=N_per_policy,
                episode_length=T,
                seed=int(seed),
                num_ref_samples=NUM_REF_SAMPLES,
                num_negatives=NUM_NEGATIVES,
                state0_b=state0_b,
                obs0_b=obs0_b,
                states_next_b=states_next_b,
                obs_prev_b=obs_prev_b,
                obs_next_b=obs_next_b,
                oracle_key=ok,
                neg_key=nk,
            )

            for i, spec in enumerate(methods):
                mk = jax.random.fold_in(mk_root, i)
                samples, dm = compute_method_samples_and_density(
                    spec,
                    env=env,
                    embedding_model=embedding_model,
                    k=k,
                    state0_b=state0_b,
                    obs0_b=obs0_b,
                    states_next_b=states_next_b,
                    obs_prev_b=obs_prev_b,
                    obs_next_b=obs_next_b,
                    neg_pts_b=neg_pts_b,
                    num_ref_samples=NUM_REF_SAMPLES,
                    key=mk,
                    oracle_samples=oracle,
                )
                for name, fn in METRIC_FNS.items():
                    arr = np.asarray(fn(samples, oracle))
                    metrics[name][spec][base : base + N_per_policy] = arr

                if dm is not None:
                    metrics['lds'][spec][base : base + N_per_policy] = np.asarray(
                        dm['log_density_score']
                    )

            mu = {
                m: float(np.nanmean(metrics['mmd_multibw'][m][base : base + N_per_policy]))
                for m in methods
            }
            tag = ', '.join(f'{m}={mu[m]:.4f}' for m in methods)
            print(f'  k={k} seed={seed} {comp}: mmd_multibw {tag}')

    return {
        'mmd': metrics['mmd'],
        'mmd_multibw': metrics['mmd_multibw'],
        'lds': metrics['lds'],
        'policy_idx': policy_idx,
        'seed_idx': seed_idx,
    }


def _save_per_step_npz(out_dir, per_k_data, methods, seeds, components, *, ks):
    save_dict = {
        'ks': np.asarray(list(ks)),
        'methods': np.asarray(methods),
        'seeds': np.asarray(seeds),
        'policy_names': np.asarray(components),
    }
    for k in ks:
        d = per_k_data[k]
        save_dict[f'k{k}/policy_idx'] = d['policy_idx']
        save_dict[f'k{k}/seed_idx'] = d['seed_idx']
        for metric in ('mmd', 'mmd_multibw', 'lds'):
            for m in methods:
                save_dict[f'k{k}/{m}/{metric}_per_step'] = d[metric][m]
    npz_path = os.path.join(out_dir, 'e2_per_step.npz')
    np.savez(npz_path, **save_dict)
    print(f'  wrote {npz_path}')


def _write_summary(out_dir, per_k_data, methods, seeds, components):
    ks = sorted(per_k_data.keys())

    summary = {m: {k: {} for k in ks} for m in methods}
    for k in ks:
        d = per_k_data[k]
        for m in methods:
            for metric in SUMMARY_METRICS:
                mean, sem = aggregate_mean_sem(d[metric][m])
                summary[m][k][metric] = {'mean': mean, 'sem': sem}

    npz_path = os.path.join(out_dir, 'e2_summary.npz')
    save = {'ks': np.asarray(ks), 'methods': np.asarray(methods)}
    for metric in SUMMARY_METRICS:
        save[f'{metric}/mean'] = np.asarray(
            [[summary[m][k][metric]['mean'] for k in ks] for m in methods]
        )
        save[f'{metric}/sem'] = np.asarray(
            [[summary[m][k][metric]['sem'] for k in ks] for m in methods]
        )
    np.savez(npz_path, **save)
    print(f'  wrote {npz_path}')

    tex_path = os.path.join(out_dir, 'e2_summary.tex')
    _write_summary_table(tex_path, summary, methods, ks, seeds, components)


def _write_summary_table(out_path, summary, methods, ks, seeds, components):
    from nbf.domains.pursuit_evasion.visualize import order_series, series_style

    ordered = order_series(methods)
    metric_order = (('mmd_multibw', True), ('lds', False))

    # Bold best per (k, metric).
    best = {}
    for metric, lower_is_better in metric_order:
        for k in ks:
            cands = [(m, summary[m][k][metric]['mean']) for m in ordered]
            cands = [(m, v) for m, v in cands if np.isfinite(v)]
            if not cands:
                continue
            cmp = min if lower_is_better else max
            best[(metric, k)] = cmp(cands, key=lambda x: x[1])[0]

    col_spec = 'l' + 'r' * len(ks)
    ncols = len(ks) + 1
    lines = ['\\begin{table}[t]', '\\centering', '\\small']
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')
    header = ['Method'] + [f'$k={k}$' for k in ks]
    lines.append(' & '.join(header) + ' \\\\')
    for _pi, (metric, _) in enumerate(metric_order):
        label = (
            'MMD$^2$ (multi-bw, lower is better)'
            if metric == 'mmd_multibw'
            else 'LDS (higher is better)'
        )
        lines.append('\\midrule')
        lines.append(f'\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{label}}}}} \\\\')
        lines.append('\\addlinespace[2pt]')
        for m in ordered:
            label_str, _ = series_style(m)
            cells = [label_str]
            for k in ks:
                d = summary[m][k][metric]
                txt = fmt_val(d['mean'])
                if np.isfinite(d['sem']) and d['sem'] > 0:
                    txt = f'{txt} $\\pm$ {fmt_val(d["sem"])}'
                if best.get((metric, k)) == m:
                    txt = f'\\textbf{{{txt}}}'
                cells.append(txt)
            lines.append(' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    n_seeds = len(seeds)
    n_pol = len(components)
    caption = (
        f'Pursuit-evasion E2 summary: multi-bandwidth MMD$^2$ (lower is better) and log-density '
        f'score (higher is better) vs gold-SMC oracle, aggregated over '
        f'{n_seeds} seed(s), {n_pol} policy classes ({", ".join(components)}), and time. '
        'Mean $\\pm$ SEM over episodes.'
    )
    lines.append(f'\\caption{{{caption}}}')
    lines.append('\\label{tab:e2_summary}')
    lines.append('\\end{table}')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  wrote {out_path}')


def _plot_per_policy_grid(out_dir, per_k_data, methods, components):
    from nbf.domains.pursuit_evasion.visualize import order_series, series_style

    ks = sorted(per_k_data.keys())
    ordered = order_series(methods)

    for metric, ylabel in PLOT_METRICS:
        n_rows = len(ks)
        n_cols = len(components)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.5 * n_cols, 3.0 * n_rows),
            squeeze=False,
            sharey='row',
        )
        for ri, k in enumerate(ks):
            d = per_k_data[k]
            policy_idx = d['policy_idx']
            for ci, comp in enumerate(components):
                ax = axes[ri][ci]
                mask = policy_idx == ci
                for m in ordered:
                    arr = d[metric].get(m)
                    if arr is None:
                        continue
                    sub = arr[mask]
                    if sub.size == 0 or np.all(np.isnan(sub)):
                        continue
                    ts = np.arange(sub.shape[1])
                    mean, sem = aggregate_per_step_mean_sem(sub)
                    label, color = series_style(m)
                    ax.plot(ts, mean, label=label, color=color, linewidth=1.4)
                    ax.fill_between(
                        ts, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0
                    )
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax.tick_params(axis='both', labelsize=10)
                if ri == 0:
                    ax.set_title(comp, fontsize=14)
                if ri == n_rows - 1:
                    ax.set_xlabel('t', fontsize=12)
                if ci == 0:
                    ax.set_ylabel(f'$k={k}$\n{ylabel}', fontsize=12)
                if metric == 'mmd_multibw':
                    ax.set_yscale('log')
        axes[0][-1].legend(fontsize=10, loc='best')
        fig.tight_layout()
        suffix = 'mmd_multibw' if metric == 'mmd_multibw' else 'lds'
        out = os.path.join(out_dir, f'e2_{suffix}_vs_time_per_policy.pdf')
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f'  wrote {out}')


def _load_per_step_npz(path):
    d = np.load(path)
    ks = [int(x) for x in d['ks']]
    methods = [str(x) for x in d['methods']]
    seeds = list(d['seeds']) if 'seeds' in d.files else [0]
    components = [str(x) for x in d['policy_names']] if 'policy_names' in d.files else []
    per_k_data = {}
    for k in ks:
        per_k_data[k] = {
            'mmd': {m: d[f'k{k}/{m}/mmd_per_step'] for m in methods},
            'mmd_multibw': {m: d[f'k{k}/{m}/mmd_multibw_per_step'] for m in methods},
            'lds': {m: d[f'k{k}/{m}/lds_per_step'] for m in methods},
            'policy_idx': d[f'k{k}/policy_idx'],
            'seed_idx': d[f'k{k}/seed_idx'],
        }
    return per_k_data, methods, seeds, components


def replot(args):
    out_dir = args.output_dir
    npz_path = os.path.join(out_dir, 'e2_per_step.npz')
    if not os.path.exists(npz_path):
        print(f'E2 replot: missing {npz_path}')
        return
    per_k_data, methods, seeds, components = _load_per_step_npz(npz_path)
    _write_summary(out_dir, per_k_data, tuple(methods), seeds, components)
    _plot_per_policy_grid(out_dir, per_k_data, tuple(methods), components)
