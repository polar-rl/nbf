import argparse
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nbf.domains.lorenz96.env import Lorenz96Env
from nbf.filters.enkf import EnsembleKalmanFilter
from nbf.filters.particle import ParticleFilter

V1_BAND = (0.15, 0.30)
V2_BAND = (0.70, 1.05)
V3_MAX_ESS_FRAC = 0.05
V3_MIN_RMSE_RATIO = 2.0


@eqx.filter_jit
def enkf_run(env, filt, model_forcing, num_steps, key):
    reset_key, filt_key, scan_key = jax.random.split(key, 3)
    state, ts = env.reset(reset_key)
    pset = filt.reset(env, state, ts.observation, filt_key)
    if model_forcing is not None:
        pset = eqx.tree_at(
            lambda p: p.particles.forcing,
            pset,
            jnp.full_like(pset.particles.forcing, model_forcing),
        )

    def step(carry, k):
        state, pset = carry
        step_key, upd_key = jax.random.split(k)
        state, ts = env.step(state, 0, step_key)
        pset, _, _, _ = filt.update(env, pset, None, None, 0, ts.observation, upd_key)
        err = jnp.mean(pset.particles.x, axis=0) - state.x
        return (state, pset), jnp.sqrt(jnp.mean(err**2))

    _, rmse = jax.lax.scan(step, (state, pset), jax.random.split(scan_key, num_steps))
    return rmse


@eqx.filter_jit
def sir_run(env, num_particles, num_steps, key):
    reset_key, filt_key, scan_key = jax.random.split(key, 3)
    state, ts = env.reset(reset_key)
    pset = ParticleFilter(num_particles=num_particles).reset(env, state, ts.observation, filt_key)

    def step(carry, k):
        state, pset = carry
        step_key, sim_key, res_key = jax.random.split(k, 3)
        state, ts = env.step(state, 0, step_key)
        pset, _ = pset.simulate(env, 0, ts.observation, sim_key)
        ess = pset.neff()
        max_w = jnp.max(pset.weights)
        mean = jnp.sum(pset.weights[:, None] * pset.particles.x, axis=0)
        rmse = jnp.sqrt(jnp.mean((mean - state.x) ** 2))
        pset = pset.systematic_resample(res_key)
        return (state, pset), (rmse, ess, max_w)

    _, (rmse, ess, max_w) = jax.lax.scan(step, (state, pset), jax.random.split(scan_key, num_steps))
    return rmse, ess, max_w


def _sweep_enkf(env, members, model_forcing, inflations, radii, num_steps, spinup, key):
    results = []
    for inflation in inflations:
        for radius in radii:
            filt = EnsembleKalmanFilter(num_members=members, inflation=inflation, loc_radius=radius)
            rmse = np.asarray(enkf_run(env, filt, model_forcing, num_steps, key))
            score = float(np.mean(rmse[spinup:])) if np.all(np.isfinite(rmse)) else float('inf')
            results.append((inflation, radius, score))
            print(f'  inflation={inflation:<5} radius={radius:<3} rmse_a={score:.3f}')
    return min(results, key=lambda r: r[2])


def run_v1(args):
    num_steps = 550 if args.quick else args.steps or 5500
    spinup = num_steps // 11
    env = Lorenz96Env(
        K=40,
        time_limit=num_steps + 1,
        sigma_dyn=0.0,
        sigma_obs_min=1.0,
        sigma_obs_max=1.0,
        f_bands=((8.0, 8.0),),
        obs_sites=tuple(range(40)),
    )
    print(
        f'V1: K=40, F=8, fully observed, R=I, window 0.05, m={args.members}, '
        f'{num_steps} steps ({spinup} spin-up)'
    )
    print('   anchor: stochastic EnKF (PertObs) rmse_a ~ 0.22, EnSRF/DEnKF ~ 0.18')
    print('   (Sakov-Oke 2008 Table 1 via DAPPER mods/Lorenz96/sakov2008)')
    inflation, radius, best = _sweep_enkf(
        env,
        args.members,
        None,
        (1.02, 1.05, 1.08, 1.12),
        (2.0, 4.0, 6.0, 8.0),
        num_steps,
        spinup,
        jax.random.PRNGKey(args.seed),
    )
    lo, hi = V1_BAND
    ok = lo <= best <= hi
    print(f'V1 best: rmse_a={best:.3f} (inflation={inflation}, radius={radius})')
    print(f'V1 {"PASS" if ok else "FAIL"}: expected [{lo}, {hi}]')
    return ok


def run_v2(args):
    num_steps = 200 if args.quick else args.steps or 1000
    spinup = num_steps // 10
    env = Lorenz96Env(
        K=40,
        time_limit=num_steps + 1,
        dt=0.05,
        substeps=6,
        sigma_dyn=0.0,
        sigma_obs_min=0.5,
        sigma_obs_max=0.5,
        f_bands=((8.0, 8.0),),
    )
    print(
        f'V2: K=40, truth F=8 / model F=9, every 2nd site, sigma_obs=0.5, '
        f'window 0.3, m={args.members}, {num_steps} steps ({spinup} spin-up)'
    )
    print('   anchor: LMCPF 0.77, LETKF 0.86, LAPF 0.97 (Schenk, Potthast & Rojahn 2022, Fig. 11)')
    inflation, radius, best = _sweep_enkf(
        env,
        args.members,
        9.0,
        (1.05, 1.12, 1.2, 1.3),
        (1.0, 2.0, 4.0),
        num_steps,
        spinup,
        jax.random.PRNGKey(args.seed),
    )
    lo, hi = V2_BAND
    ok = lo <= best <= hi
    print(f'V2 best: rmse_a={best:.3f} (inflation={inflation}, radius={radius})')
    print(f'V2 {"PASS" if ok else "FAIL"}: expected [{lo}, {hi}] (band, not digit)')
    return ok


def run_v3(args):
    num_steps = 100 if args.quick else args.steps or 500
    spinup = num_steps // 10
    env = Lorenz96Env(
        K=40,
        time_limit=num_steps + 1,
        sigma_dyn=0.25,
        sigma_obs_min=1.0,
        sigma_obs_max=1.0,
        f_bands=((8.0, 8.0),),
    )
    key = jax.random.PRNGKey(args.seed)
    print(f'V3: K=40, F=8, odd sites, sigma_obs=1, sigma_dyn=0.25, {num_steps} steps')

    filt = EnsembleKalmanFilter(num_members=args.members, inflation=1.05, loc_radius=4.0)
    enkf_rmse = np.asarray(enkf_run(env, filt, None, num_steps, key))
    enkf_score = float(np.mean(enkf_rmse[spinup:]))
    print(f'  EnKF(m={args.members}) reference: rmse_a={enkf_score:.3f}')

    ok = True
    for n in (1024, 8192):
        rmse, ess, max_w = (np.asarray(a) for a in sir_run(env, n, num_steps, key))
        score = float(np.mean(rmse[spinup:]))
        ess_frac = float(np.median(ess / n))
        collapsed = ess_frac < V3_MAX_ESS_FRAC and score > V3_MIN_RMSE_RATIO * enkf_score
        ok = ok and collapsed
        print(
            f'  SIR(n={n}): rmse_a={score:.3f} median ESS/n={ess_frac:.4f} '
            f'mean max_w={float(np.mean(max_w)):.3f} '
            f'{"collapse visible" if collapsed else "NO COLLAPSE — suspect weights"}'
        )
    print(f'V3 {"PASS" if ok else "FAIL"}: SIR must show weight collapse at K=40')
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description='L96 external validation anchors (spec Sec 4.6).')
    parser.add_argument('anchor', choices=['v1', 'v2', 'v3', 'all'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--members', type=int, default=20)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    runners = {'v1': run_v1, 'v2': run_v2, 'v3': run_v3}
    names = ('v1', 'v2', 'v3') if args.anchor == 'all' else (args.anchor,)
    ok = all([runners[name](args) for name in names])
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
