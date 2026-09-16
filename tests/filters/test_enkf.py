import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nbf.domains.lorenz96.env import Lorenz96Env, rk4_window
from nbf.filters.enkf import (
    EnsembleKalmanFilter,
    LocalEnsembleTransformKalmanFilter,
    gaspari_cohn,
    ring_taper,
)

K = 6
SIGMA_DYN = 0.3
SIGMA_OBS = 0.8
SIGMA_INIT = 1.5
T = 5
OBS_IDX = np.arange(1, K, 2)


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def env():
    return Lorenz96Env(
        K=K,
        time_limit=T,
        sigma_dyn=SIGMA_DYN,
        sigma_obs_min=SIGMA_OBS,
        sigma_obs_max=SIGMA_OBS,
        f_bands=((7.9, 8.1),),
        spinup_time=2.0,
        sigma_init=SIGMA_INIT,
        linear=True,
    )


def _linear_map(env, forcing):
    b = np.asarray(rk4_window(jnp.zeros(K), forcing, env.dt, env.substeps, linear=True))
    a_vec = np.asarray(rk4_window(jnp.ones(K), forcing, env.dt, env.substeps, linear=True)) - b
    return float(a_vec[0]), b


def _kalman_moments(env, forcing, center, obs_values):
    a, b = _linear_map(env, forcing)
    H = np.zeros((K // 2, K))
    for i, j in enumerate(OBS_IDX):
        H[i, j] = 1.0
    mu = np.asarray(center, dtype=np.float64)
    P = SIGMA_INIT**2 * np.eye(K)
    mus, Ps = [], []
    for y in np.asarray(obs_values, dtype=np.float64):
        mu = a * mu + b
        P = a**2 * P + SIGMA_DYN**2 * np.eye(K)
        S = H @ P @ H.T + SIGMA_OBS**2 * np.eye(K // 2)
        G = P @ H.T @ np.linalg.inv(S)
        mu = mu + G @ (y - H @ mu)
        P = (np.eye(K) - G @ H) @ P
        mus.append(mu.copy())
        Ps.append(P.copy())
    return np.stack(mus), np.stack(Ps)


class TestGaspariCohn:
    def test_endpoint_values(self):
        r = jnp.array([0.0, 1.0, 2.0, 2.5])
        got = np.asarray(gaspari_cohn(r))
        assert got[0] == pytest.approx(1.0)
        assert got[1] == pytest.approx(5.0 / 24.0, abs=1e-6)
        assert got[2] == pytest.approx(0.0, abs=1e-6)
        assert got[3] == 0.0

    def test_continuous_and_decreasing(self):
        r = jnp.linspace(0.0, 2.2, 221)
        vals = np.asarray(gaspari_cohn(r))
        assert np.all(np.diff(vals) <= 1e-6)
        assert np.all(vals >= 0.0)
        assert np.all(vals <= 1.0)

    def test_ring_taper_wraps(self):
        taper = np.asarray(ring_taper(8, jnp.arange(8), 2.0))
        assert np.allclose(np.diag(taper), 1.0)
        assert taper[0, 7] == pytest.approx(taper[0, 1])
        assert taper[0, 4] == pytest.approx(0.0, abs=1e-6)


FILTERS = {
    'enkf': EnsembleKalmanFilter,
    'letkf': LocalEnsembleTransformKalmanFilter,
}


class TestLinearModeKalman:
    @pytest.mark.parametrize('name', list(FILTERS))
    def test_matches_exact_kalman(self, env, key, name):
        reset_key, filt_key = jax.random.split(key)
        state, ts = env.reset(reset_key)
        enkf = FILTERS[name](num_members=4096, inflation=1.0, loc_radius=1e6)
        pset = enkf.reset(env, state, ts.observation, filt_key)

        obs_values = []
        members = []
        for sub in jax.random.split(filt_key, T):
            step_key, upd_key = jax.random.split(sub)
            state, ts = env.step(state, 0, step_key)
            pset, _, _, _ = enkf.update(env, pset, None, None, 0, ts.observation, upd_key)
            obs_values.append(np.asarray(ts.observation.values))
            members.append(np.asarray(pset.particles.x))

        mus, Ps = _kalman_moments(
            env, float(state.forcing), np.asarray(state.center), np.stack(obs_values)
        )
        for t in range(T):
            X = members[t].astype(np.float64)
            assert np.allclose(X.mean(axis=0), mus[t], atol=0.15), t
            assert np.allclose(X.var(axis=0), np.diag(Ps[t]), rtol=0.3, atol=0.02), t

    def test_forecast_carries_forcing(self, env, key):
        state, ts = env.reset(key)
        enkf = EnsembleKalmanFilter(num_members=32, inflation=1.0, loc_radius=1e6)
        pset = enkf.reset(env, state, ts.observation, key)
        assert np.allclose(np.asarray(pset.particles.forcing), float(state.forcing))
        assert np.allclose(pset.weights, 1.0 / 32)


class TestLocalizedTracking:
    @pytest.mark.parametrize('name', list(FILTERS))
    def test_k40_fully_observed_tracks_truth(self, key, name):
        env = Lorenz96Env(
            K=40,
            time_limit=250,
            sigma_dyn=0.0,
            sigma_obs_min=1.0,
            sigma_obs_max=1.0,
            f_bands=((8.0, 8.0),),
            obs_sites=tuple(range(40)),
        )
        enkf = FILTERS[name](num_members=20, inflation=1.05, loc_radius=4.0)
        reset_key, scan_key = jax.random.split(key)
        state, ts = env.reset(reset_key)
        pset = enkf.reset(env, state, ts.observation, reset_key)

        def step(carry, k):
            state, pset = carry
            step_key, upd_key = jax.random.split(k)
            state, ts = env.step(state, 0, step_key)
            pset, _, _, _ = enkf.update(env, pset, None, None, 0, ts.observation, upd_key)
            err = jnp.mean(pset.particles.x, axis=0) - state.x
            return (state, pset), jnp.sqrt(jnp.mean(err**2))

        _, rmse = jax.lax.scan(step, (state, pset), jax.random.split(scan_key, 250))
        assert np.all(np.isfinite(np.asarray(rmse)))
        assert float(jnp.mean(rmse[100:])) < 1.0


class TestLETKF:
    def test_repr(self):
        assert repr(LocalEnsembleTransformKalmanFilter(num_members=7)) == 'LETKF (7)'

    def test_analysis_is_deterministic(self, env, key):
        state, ts = env.reset(key)
        letkf = LocalEnsembleTransformKalmanFilter(num_members=64, inflation=1.0, loc_radius=1e6)
        pset = letkf.reset(env, state, ts.observation, key)
        state, ts = env.step(state, 0, key)
        X = env.state_array(pset.particles)
        k1, k2 = jax.random.split(key)
        a = letkf._analysis(X, ts.observation, env.obs_idx, k1)
        b = letkf._analysis(X, ts.observation, env.obs_idx, k2)
        assert np.allclose(np.asarray(a), np.asarray(b))
        assert not np.allclose(np.asarray(a), np.asarray(X))
