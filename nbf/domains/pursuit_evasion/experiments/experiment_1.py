import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nbf.domains.pursuit_evasion.env import L, normalize_positions
from nbf.domains.pursuit_evasion.eval.base import make_env_kn, make_pe_gmf
from nbf.domains.pursuit_evasion.policies import make_goal_policy
from nbf.domains.pursuit_evasion.visualize import plot_e1_heatmaps
from nbf.eval.base import (
    env_rollout_batch,
    parse_csv_methods,
    parse_method_spec,
    slice_rollout_no_gt,
)
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.particle import ParticleFilter

DEFAULT_METHODS = ('pf_oracle', 'gmf_4', 'gmf_16', 'gmf_64', 'pf_256', 'nbf_256')
DEFAULT_TIMESTEPS = (1, 5, 10, 15, 20)
COMPACT_TIMESTEPS = (5, 10)
GRID_RES = 64


def _make_grid(res: int):
    xs = jnp.linspace(0.0, L, res)
    ys = jnp.linspace(0.0, L, res)
    XX, YY = jnp.meshgrid(xs, ys, indexing='xy')
    pts = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)  # (res*res, 2)
    return XX, YY, pts


@eqx.filter_jit
def _nbf_grid_logp(model, embedding, pts: jnp.ndarray):
    pts_norm = normalize_positions(pts.reshape(-1, 1, 2)).reshape(-1, 2)
    out = model.log_prob(pts_norm, embedding)
    return out[0] if isinstance(out, tuple) else out


@eqx.filter_jit
def _pf_grid_logp(parts_pos, weights, grid_res: int):
    parts = parts_pos[:, 0, :]  # (P, 2) — k=1
    cell = L / grid_res
    ix = jnp.clip((parts[:, 0] / cell).astype(jnp.int32), 0, grid_res - 1)
    iy = jnp.clip((parts[:, 1] / cell).astype(jnp.int32), 0, grid_res - 1)
    flat = iy * grid_res + ix
    w = weights / (jnp.sum(weights) + 1e-30)
    pmf = jnp.zeros(grid_res * grid_res).at[flat].add(w)
    return jnp.log(pmf + 1e-30)


@eqx.filter_jit
def _gmf_grid_logp(weights, mus, sigmas, pts: jnp.ndarray):
    safe_w = weights / (jnp.sum(weights) + 1e-30)
    log_w = jnp.log(safe_w + 1e-30)
    n = mus.shape[-1]

    def comp_logpdf(mu, sig, x):
        sig_sym = 0.5 * (sig + sig.T) + 1e-6 * jnp.eye(n)
        L_chol = jnp.linalg.cholesky(sig_sym)
        diff = x - mu
        sol = jax.scipy.linalg.solve_triangular(L_chol, diff, lower=True)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L_chol)))
        return -0.5 * (jnp.sum(sol * sol) + log_det + n * jnp.log(2.0 * jnp.pi))

    def logpdf(x):
        per_comp = jax.vmap(lambda mu, sig: comp_logpdf(mu, sig, x))(mus, sigmas)
        return jax.scipy.special.logsumexp(log_w + per_comp)

    return jax.vmap(logpdf)(pts)


def _run_filter_capture(
    spec: str,
    env,
    embedding_model,
    state0,
    obs0,
    states_next,
    obs_prev,
    obs_next,
    key,
    timesteps,
):
    kind, n = parse_method_spec(spec)
    T = obs_prev.ranges.shape[0]
    sk = jax.random.split(key, T + 2)

    if kind == 'nbf':
        nbf = NeuralBayesianFilter(embedding_model, num_particles=n)
        embedding = nbf.reset(env, state0, obs0, sk[0])
        captured = {}
        for t in range(T):
            embedding, _ = nbf.update(
                env,
                embedding,
                jax.tree.map(lambda x, t=t: x[t], states_next),
                jax.tree.map(lambda x, t=t: x[t], obs_prev),
                0,
                jax.tree.map(lambda x, t=t: x[t], obs_next),
                sk[t + 1],
            )
            if t in timesteps:
                captured[t] = ('nbf', embedding)
        return captured

    if kind == 'pf' or kind == 'pf_oracle':
        from nbf.eval.dispatch import ORACLE_PF_COUNT

        pf_n = ORACLE_PF_COUNT if kind == 'pf_oracle' else n
        pf = ParticleFilter(num_particles=pf_n)
        pf_set = pf.reset(env, state0, obs0, sk[0])
        captured = {}
        for t in range(T):
            pf_set, _ = pf.update(
                env,
                pf_set,
                jax.tree.map(lambda x, t=t: x[t], states_next),
                jax.tree.map(lambda x, t=t: x[t], obs_prev),
                0,
                jax.tree.map(lambda x, t=t: x[t], obs_next),
                sk[t + 1],
            )
            if t in timesteps:
                captured[t] = ('pf', (pf_set.particles.positions, pf_set.weights, pf_n))
        return captured

    if kind == 'gmf':
        gmf = make_pe_gmf(num_evaders=env.num_evaders, M=n)
        w, mu, sg = gmf.init(key=sk[0])
        captured = {}
        for t in range(T):
            obs_t = jax.tree.map(lambda x, t=t: x[t], obs_next)
            w, mu, sg = gmf.step(w, mu, sg, env.policy, env.obstacles, obs_t)
            if t in timesteps:
                captured[t] = ('gmf', (w, mu, sg))
        return captured

    raise ValueError(spec)


def _grid_logp_for(captured_t, model, pts):
    kind, payload = captured_t
    if kind == 'nbf':
        return _nbf_grid_logp(model, payload, pts)
    if kind == 'pf':
        positions, weights, _P = payload
        return _pf_grid_logp(positions, weights, GRID_RES)
    if kind == 'gmf':
        w, mu, sg = payload
        return _gmf_grid_logp(w, mu, sg, pts)
    raise ValueError(kind)


def run(args, embedding_model, key):
    if args.num_evaders != 1:
        print('E1 requires k=1; skipping.')
        return

    # E1 always uses a fixed seed for the heatmap figure (chosen episode).
    del key
    key = jax.random.PRNGKey(3)

    methods = parse_csv_methods(args, 'e1_methods', DEFAULT_METHODS)
    # 1-indexed display timesteps; internally we slice rollout arrays at index t-1.
    timesteps = tuple(t for t in DEFAULT_TIMESTEPS if t <= args.episode_length)
    indices = tuple(t - 1 for t in timesteps)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    policy = make_goal_policy(jnp.array([9.0, 2.0]), concentration=3.0)
    env = make_env_kn(1, args.episode_length, (policy,))

    rk, fk = jax.random.split(key)
    state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = _rollout_one_episode(
        env, args.episode_length, rk
    )

    XX, YY, pts = _make_grid(GRID_RES)

    all_grids = {}
    for spec in methods:
        captured = _run_filter_capture(
            spec,
            env,
            embedding_model,
            state0_b,
            obs0_b,
            states_next_b,
            obs_prev_b,
            obs_next_b,
            fk,
            indices,
        )
        per_t = []
        for idx in indices:
            logp = _grid_logp_for(captured[idx], embedding_model, pts)
            per_t.append(np.asarray(logp.reshape(GRID_RES, GRID_RES)))
        all_grids[spec] = np.stack(per_t, axis=0)

    true_pos = np.asarray(states_next_b.positions[list(indices), 0])
    full_traj = np.asarray(states_next_b.positions[:, 0])
    npz_path = os.path.join(out_dir, 'e1_k1_heatmaps.npz')
    np.savez(
        npz_path,
        xs=np.asarray(XX[0]),
        ys=np.asarray(YY[:, 0]),
        timesteps=np.asarray(timesteps),
        true_positions=true_pos,
        full_trajectory=full_traj,
        **{k: v for k, v in all_grids.items()},
    )
    print(f'  wrote {npz_path}')
    pdf_path = os.path.join(out_dir, 'e1_k1_heatmaps.pdf')
    plot_e1_heatmaps(all_grids, true_pos, timesteps, pdf_path, compact=False, trajectory=full_traj)
    print(f'  wrote {pdf_path}')

    compact_rows = [i for i, t in enumerate(timesteps) if t in COMPACT_TIMESTEPS]
    if compact_rows:
        compact_grids = {k: v[compact_rows] for k, v in all_grids.items()}
        compact_true = true_pos[compact_rows]
        compact_ts = tuple(timesteps[i] for i in compact_rows)
        pdf_compact = os.path.join(out_dir, 'e1_k1_heatmaps_compact.pdf')
        plot_e1_heatmaps(
            compact_grids, compact_true, compact_ts, pdf_compact, compact=True, trajectory=full_traj
        )
        print(f'  wrote {pdf_compact}')


def _rollout_one_episode(env, episode_length, key):
    state0_all, obs0_all = env_rollout_batch(env, episode_length, key[None])
    obs_prev_b, obs_next_b, states_next_b, state0_b, obs0_b = slice_rollout_no_gt(
        state0_all, obs0_all
    )
    state0 = jax.tree.map(lambda x: x[0], state0_b)
    obs0 = jax.tree.map(lambda x: x[0], obs0_b)
    states_next = jax.tree.map(lambda x: x[0], states_next_b)
    obs_prev = jax.tree.map(lambda x: x[0], obs_prev_b)
    obs_next = jax.tree.map(lambda x: x[0], obs_next_b)
    return state0, obs0, states_next, obs_prev, obs_next


def replot(args):
    out_dir = args.output_dir
    npz_path = os.path.join(out_dir, 'e1_k1_heatmaps.npz')
    d = np.load(npz_path)
    timesteps = tuple(int(t) for t in d['timesteps'])
    true_pos = d['true_positions']
    full_traj = d['full_trajectory'] if 'full_trajectory' in d.files else None
    reserved = {'xs', 'ys', 'timesteps', 'true_positions', 'full_trajectory'}
    grids = {k: d[k] for k in d.files if k not in reserved}
    pdf_path = os.path.join(out_dir, 'e1_k1_heatmaps.pdf')
    plot_e1_heatmaps(grids, true_pos, timesteps, pdf_path, compact=False, trajectory=full_traj)
    print(f'  wrote {pdf_path}')
    compact_rows = [i for i, t in enumerate(timesteps) if t in COMPACT_TIMESTEPS]
    if compact_rows:
        compact_grids = {k: v[compact_rows] for k, v in grids.items()}
        compact_true = true_pos[compact_rows]
        compact_ts = tuple(timesteps[i] for i in compact_rows)
        pdf_compact = os.path.join(out_dir, 'e1_k1_heatmaps_compact.pdf')
        plot_e1_heatmaps(
            compact_grids,
            compact_true,
            compact_ts,
            pdf_compact,
            compact=True,
            trajectory=full_traj,
        )
        print(f'  wrote {pdf_compact}')
