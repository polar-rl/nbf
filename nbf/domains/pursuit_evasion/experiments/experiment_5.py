import os

import jax
import numpy as np

from nbf.domains.pursuit_evasion.eval.base import (
    make_env_kn,
    parse_csv_methods,
    rollout_episodes,
)
from nbf.domains.pursuit_evasion.eval.dispatch import NUM_REF_SAMPLES
from nbf.domains.pursuit_evasion.eval.timing import time_method_walltime
from nbf.domains.pursuit_evasion.experiments._tex import fmt_or_dash, latex_escape
from nbf.domains.pursuit_evasion.policies import make_random_walk_policy

DEFAULT_METHODS = (
    'nbf_128',
    'nbf_256',
    'nbf_1024',
    'nbf_2048',
    'nbf_8192',
    'pf_1024',
    'pf_2048',
    'pf_8192',
)
DEFAULT_NUM_REPEATS = 10
DEFAULT_NUM_WARMUP = 10


def run(args, embedding_model, key):
    k = args.num_evaders
    methods = parse_csv_methods(args, 'e5_methods', DEFAULT_METHODS)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Use a small N and short T to keep total wall time tractable. The user
    # may override --num_episodes / --episode_length.
    n_episodes = getattr(args, 'e5_num_episodes', max(args.num_episodes, 4))
    n_episodes = min(n_episodes, args.num_episodes if args.num_episodes else 8)
    episode_length = args.episode_length
    num_warmup = getattr(args, 'e5_num_warmup', DEFAULT_NUM_WARMUP)
    num_repeats = getattr(args, 'e5_num_repeats', DEFAULT_NUM_REPEATS)

    policy = make_random_walk_policy()
    env = make_env_kn(k, episode_length, (policy,) * k)
    rk, mk_root = jax.random.split(key)
    state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = rollout_episodes(
        env,
        episode_length,
        n_episodes,
        rk,
    )
    N = state0_b.positions.shape[0]
    T = obs_prev_b.ranges.shape[1]
    n_steps = N * T

    results = {}
    for i, spec in enumerate(methods):
        mk = jax.random.fold_in(mk_root, i)
        try:
            timing = time_method_walltime(
                spec,
                env=env,
                embedding_model=embedding_model,
                state0_b=state0_b,
                obs0_b=obs0_b,
                states_next_b=states_next_b,
                obs_prev_b=obs_prev_b,
                obs_next_b=obs_next_b,
                num_ref_samples=NUM_REF_SAMPLES,
                num_warmup=num_warmup,
                num_repeats=num_repeats,
                key=mk,
            )
        except Exception as e:
            print(f'  {spec}: skipped ({e})')
            continue
        timing['per_step_ms'] = 1000.0 * timing['mean_s'] / max(n_steps, 1)
        if 'filter_mean_s' in timing:
            timing['filter_per_step_ms'] = 1000.0 * timing['filter_mean_s'] / max(n_steps, 1)
            timing['sample_per_step_ms'] = 1000.0 * timing['sample_mean_s'] / max(n_steps, 1)
        timing['n_steps'] = n_steps
        results[spec] = timing
        if 'sample_per_step_ms' in timing:
            print(
                f'  {spec}: {timing["mean_s"]:.3f}s '
                f'({timing["per_step_ms"]:.2f} ms/step, '
                f'sample {timing["sample_per_step_ms"]:.2f}) '
                f'over {n_steps} steps'
            )
        else:
            print(
                f'  {spec}: {timing["mean_s"]:.3f}s '
                f'({timing["per_step_ms"]:.2f} ms/step over {n_steps} steps)'
            )

    npz_path = os.path.join(out_dir, 'e5_walltime.npz')
    specs = list(results.keys())
    np.savez(
        npz_path,
        methods=np.asarray(specs),
        mean_s=np.asarray([results[s]['mean_s'] for s in specs]),
        std_s=np.asarray([results[s]['std_s'] for s in specs]),
        per_step_ms=np.asarray([results[s]['per_step_ms'] for s in specs]),
        filter_mean_s=np.asarray([results[s].get('filter_mean_s', np.nan) for s in specs]),
        filter_std_s=np.asarray([results[s].get('filter_std_s', np.nan) for s in specs]),
        filter_per_step_ms=np.asarray(
            [results[s].get('filter_per_step_ms', np.nan) for s in specs]
        ),
        sample_mean_s=np.asarray([results[s].get('sample_mean_s', np.nan) for s in specs]),
        sample_std_s=np.asarray([results[s].get('sample_std_s', np.nan) for s in specs]),
        sample_per_step_ms=np.asarray(
            [results[s].get('sample_per_step_ms', np.nan) for s in specs]
        ),
        n_episodes=n_episodes,
        episode_length=episode_length,
        k=k,
    )
    print(f'  wrote {npz_path}')

    _write_tables(
        list(results.keys()),
        np.asarray([results[s]['mean_s'] for s in results]),
        np.asarray([results[s]['std_s'] for s in results]),
        np.asarray([results[s]['per_step_ms'] for s in results]),
        np.asarray([results[s].get('filter_per_step_ms', np.nan) for s in results]),
        np.asarray([results[s].get('sample_per_step_ms', np.nan) for s in results]),
        k,
        out_dir,
    )
    return results


def _write_tables(specs, mean_s, std_s, per_step_ms, filter_ms, sample_ms, k, out_dir):
    from nbf.domains.pursuit_evasion.visualize import order_series, series_style

    ordered = order_series(list(specs))
    idx = {s: i for i, s in enumerate(specs)}
    name_w = max(len('method'), max(len(series_style(s)[0]) for s in ordered))
    lines = [
        f'wall-clock per filter step (k={k})',
        f'{"method".ljust(name_w)}  {"mean_s":>10}  {"std_s":>10}  {"ms/step":>10}  {"sample":>10}',
    ]
    for s in ordered:
        i = idx[s]
        label = series_style(s)[0]
        lines.append(
            f'{label.ljust(name_w)}  {mean_s[i]:>10.4f}  {std_s[i]:>10.4f}  '
            f'{per_step_ms[i]:>10.3f}  '
            f'{fmt_or_dash(sample_ms[i]):>10}'
        )
    text = '\n'.join(lines) + '\n'
    txt_path = os.path.join(out_dir, 'e5_walltime.txt')
    with open(txt_path, 'w') as f:
        f.write(text)
    print(text, end='')
    print(f'  wrote {txt_path}')

    tex_lines = [
        r'\begin{table}[t]',
        r'  \centering',
        rf'  \caption{{Wall-clock per filter step (k={k}). Mean $\pm$ std over repeats. '
        r'For NBF, sample ms is the flow ODE cost; implied update ms = ms/step $-$ sample ms.}}',
        rf'  \label{{tab:e5_walltime_k{k}}}',
        r'  \begin{tabular}{lrrrr}',
        r'    \toprule',
        r'    Method & Mean (s) & Std (s) & ms / step & Sample ms \\',
        r'    \midrule',
    ]
    for s in ordered:
        i = idx[s]
        label = latex_escape(series_style(s)[0])
        tex_lines.append(
            f'    {label} & {mean_s[i]:.4f} & {std_s[i]:.4f} & {per_step_ms[i]:.3f} & '
            f'{fmt_or_dash(sample_ms[i])} \\\\'
        )
    tex_lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}', '']
    tex_path = os.path.join(out_dir, 'e5_walltime.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines))
    print(f'  wrote {tex_path}')


def replot(args):
    out_dir = args.output_dir
    npz_path = os.path.join(out_dir, 'e5_walltime.npz')
    d = np.load(npz_path, allow_pickle=False)
    specs = [str(s) for s in d['methods']]
    mean_s = d['mean_s']
    std_s = d['std_s']
    per_step_ms = d['per_step_ms']
    filter_ms = (
        d['filter_per_step_ms'] if 'filter_per_step_ms' in d.files else np.full(len(specs), np.nan)
    )
    sample_ms = (
        d['sample_per_step_ms'] if 'sample_per_step_ms' in d.files else np.full(len(specs), np.nan)
    )
    k = int(d['k']) if 'k' in d.files else args.num_evaders
    _write_tables(specs, mean_s, std_s, per_step_ms, filter_ms, sample_ms, k, out_dir)
