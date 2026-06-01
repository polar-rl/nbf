import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nbf.domains.pursuit_evasion.env import (
    SENSOR_POS,
    PursuitEvasionEnv,
    PursuitEvasionObservation,
)
from nbf.domains.pursuit_evasion.eval.base import (
    parse_csv_methods,
    rollout_episodes,
)
from nbf.domains.pursuit_evasion.eval.dispatch import (
    NUM_REF_SAMPLES,
    cache_path_for,
    compute_method_samples,
    get_or_compute_oracle,
)
from nbf.domains.pursuit_evasion.eval.metrics import mmd2_per_step_batch
from nbf.domains.pursuit_evasion.policies import make_random_walk_policy

DEFAULT_SIGMAS = (0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0)
DEFAULT_METHODS = (
    'nbf_128',
    'nbf_512',
    'nbf_1024',
    'pf_512',
    'pf_2048',
    'pf_8192',
)


def run(args, embedding_model, key):
    k = args.num_evaders
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    sigma_data = _run_sigma_sweep(args, embedding_model, k, key, out_dir)
    return sigma_data


def _override_sigma(state0_b, obs0_b, states_next_b, sigma: float, key):
    sigma_arr = jnp.asarray(sigma, dtype=jnp.float32)
    state0_b = eqx.tree_at(
        lambda s: s.observation_sigma,
        state0_b,
        jnp.broadcast_to(sigma_arr, state0_b.observation_sigma.shape),
    )
    states_next_b = eqx.tree_at(
        lambda s: s.observation_sigma,
        states_next_b,
        jnp.broadcast_to(sigma_arr, states_next_b.observation_sigma.shape),
    )
    obs0_b = eqx.tree_at(
        lambda o: o.observation_sigma,
        obs0_b,
        jnp.broadcast_to(sigma_arr, obs0_b.observation_sigma.shape),
    )
    true_ranges = jnp.linalg.norm(states_next_b.positions - SENSOR_POS, axis=-1)
    noise = jax.random.normal(key, true_ranges.shape)
    new_ranges = true_ranges + sigma_arr * noise
    sigma_obs_field = jnp.broadcast_to(sigma_arr, true_ranges.shape[:-1])
    obs_next_b = PursuitEvasionObservation(
        ranges=new_ranges,
        observation_sigma=sigma_obs_field,
    )
    sentinel = obs0_b.ranges
    prev_ranges = jnp.concatenate([sentinel[:, None, :], new_ranges[:, :-1, :]], axis=1)
    obs_prev_b = PursuitEvasionObservation(
        ranges=prev_ranges,
        observation_sigma=sigma_obs_field,
    )
    return state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b


def _run_sigma_sweep(args, embedding_model, k, key, out_dir):
    methods = parse_csv_methods(args, 'e4_sigma_methods', DEFAULT_METHODS)
    sigmas_attr = getattr(args, 'e4_sigmas', '')
    if sigmas_attr:
        sigmas = tuple(float(x.strip()) for x in sigmas_attr.split(',') if x.strip())
    else:
        sigmas = DEFAULT_SIGMAS

    # One rollout, reused across sigma values (positions are independent of obs noise).
    policy = make_random_walk_policy()
    env = PursuitEvasionEnv(
        num_evaders=k,
        time_limit=args.episode_length,
        policy=(policy,) * k,
        shuffle_observations=True,
    )
    rk, mk_root = jax.random.split(key)
    state0_b, obs0_b, states_next_b, _, _ = rollout_episodes(
        env,
        args.episode_length,
        args.num_episodes,
        rk,
    )

    out = {m: [] for m in methods}
    for j, sigma in enumerate(sigmas):
        ok, nk, mk_scale = jax.random.split(jax.random.fold_in(mk_root, j), 3)
        s0_s, o0_s, sn_s, op_s, on_s = _override_sigma(state0_b, obs0_b, states_next_b, sigma, nk)
        sigma_tag = f'{sigma:g}'.replace('.', 'p')
        cache = cache_path_for(
            out_dir,
            k,
            f'sigma_fixed_{sigma_tag}',
            args.num_episodes,
            args.episode_length,
            NUM_REF_SAMPLES,
            args.seed,
        )
        oracle = get_or_compute_oracle(
            cache_path=cache,
            env=env,
            state0_b=s0_s,
            obs0_b=o0_s,
            states_next_b=sn_s,
            obs_prev_b=op_s,
            obs_next_b=on_s,
            num_ref_samples=NUM_REF_SAMPLES,
            key=ok,
        )
        for i, spec in enumerate(methods):
            mk = jax.random.fold_in(mk_scale, i)
            samples = compute_method_samples(
                spec,
                env=env,
                embedding_model=embedding_model,
                state0_b=s0_s,
                obs0_b=o0_s,
                states_next_b=sn_s,
                obs_prev_b=op_s,
                obs_next_b=on_s,
                num_ref_samples=NUM_REF_SAMPLES,
                key=mk,
            )
            mmd_ps = np.asarray(mmd2_per_step_batch(samples, oracle))
            out[spec].append(
                {
                    'sigma': float(sigma),
                    'mmd_mean': float(mmd_ps.mean()),
                    'mmd_sem': float(mmd_ps.mean(axis=1).std() / np.sqrt(mmd_ps.shape[0])),
                }
            )
            print(f'  sigma={sigma:.4f} {spec}: MMD^2={mmd_ps.mean():.4f}')

    npz_path = os.path.join(out_dir, 'e4_sigma_sweep.npz')
    np.savez(
        npz_path,
        sigmas=np.asarray(sigmas),
        **{f'{m}/mmd_mean': np.asarray([x['mmd_mean'] for x in out[m]]) for m in methods},
        **{f'{m}/mmd_sem': np.asarray([x['mmd_sem'] for x in out[m]]) for m in methods},
    )
    print(f'  wrote {npz_path}')

    _plot_sigma_sweep(
        sigmas,
        {m: np.asarray([x['mmd_mean'] for x in out[m]]) for m in methods},
        {m: np.asarray([x['mmd_sem'] for x in out[m]]) for m in methods},
        os.path.join(out_dir, 'e4_sigma_sweep.pdf'),
    )
    return out


def _plot_sigma_sweep(sigmas, means, sems, pdf_path):
    from nbf.domains.pursuit_evasion.visualize import series_style

    method_names = list(means.keys())
    n_methods = len(method_names)
    n_sigmas = len(sigmas)
    x = np.arange(n_sigmas)
    total_width = 0.8
    bar_w = total_width / max(n_methods, 1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, m in enumerate(method_names):
        label, color = series_style(m)
        offset = (i - (n_methods - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            means[m],
            width=bar_w,
            yerr=sems[m],
            label=label,
            color=color,
            capsize=2,
            edgecolor='black',
            linewidth=0.3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s:g}' for s in sigmas], fontsize=12)
    ax.set_xlabel(r'$\sigma_r$', fontsize=12)
    ax.set_ylabel('MMD$^2$', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(fontsize=10, ncol=2)
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=120)
    plt.close(fig)
    print(f'  wrote {pdf_path}')


def replot(args):
    out_dir = args.output_dir
    sigma_npz = os.path.join(out_dir, 'e4_sigma_sweep.npz')
    if os.path.exists(sigma_npz):
        d = np.load(sigma_npz)
        sigmas = d['sigmas'] if 'sigmas' in d.files else d['scales']
        methods = []
        means: dict = {}
        sems: dict = {}
        for key in d.files:
            if key.endswith('/mmd_mean'):
                m = key[: -len('/mmd_mean')]
                methods.append(m)
                means[m] = d[key]
                sems[m] = d[f'{m}/mmd_sem']
        _plot_sigma_sweep(sigmas, means, sems, os.path.join(out_dir, 'e4_sigma_sweep.pdf'))
