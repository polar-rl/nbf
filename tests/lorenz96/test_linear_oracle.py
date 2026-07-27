import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nbf.domains.lorenz96.env import Lorenz96Env, rk4_window
from nbf.domains.lorenz96.eval_utils import (
    compute_l96_gold_smc_samples,
    compute_method_samples,
)
from nbf.domains.lorenz96.generate_data import scenario_block
from nbf.eval.base import rollout_episodes

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


def _check_samples_against_kf(samples_NTSK, state0_b, obs_next_b, env, mean_atol, var_rtol):
    for i in range(samples_NTSK.shape[0]):
        mus, Ps = _kalman_moments(
            env,
            float(np.asarray(state0_b.forcing)[i]),
            np.asarray(state0_b.center)[i],
            np.asarray(obs_next_b.values)[i],
        )
        samps = np.asarray(samples_NTSK[i], dtype=np.float64)
        for t in range(T):
            assert np.allclose(samps[t].mean(axis=0), mus[t], atol=mean_atol), (i, t)
            assert np.allclose(samps[t].var(axis=0), np.diag(Ps[t]), rtol=var_rtol, atol=0.02), (
                i,
                t,
            )


class TestLinearModeOracle:
    @pytest.mark.parametrize(
        ('spec', 'mean_atol', 'var_rtol'),
        [
            ('pf_16384', 0.15, 0.35),
            ('pf_reg_16384', 0.15, 0.35),
            ('pf_rm_16384', 0.15, 0.35),
        ],
    )
    def test_filters_through_dispatch(self, env, key, spec, mean_atol, var_rtol):
        rollout_key, method_key = jax.random.split(key)
        state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = rollout_episodes(
            env, T, 2, rollout_key
        )
        samples = compute_method_samples(
            spec,
            env=env,
            embedding_model=None,
            state0_b=state0_b,
            obs0_b=obs0_b,
            states_next_b=states_next_b,
            obs_prev_b=obs_prev_b,
            obs_next_b=obs_next_b,
            num_ref_samples=2048,
            key=method_key,
        )
        _check_samples_against_kf(
            np.asarray(samples), state0_b, obs_next_b, env, mean_atol=mean_atol, var_rtol=var_rtol
        )

    def test_gold_smc_eval_stream(self, env, key):
        rollout_key, oracle_key = jax.random.split(key)
        state0_b, _, _, _, obs_next_b = rollout_episodes(env, T, 2, rollout_key)
        samples = compute_l96_gold_smc_samples(env, 32768, 4096, state0_b, obs_next_b, oracle_key)
        _check_samples_against_kf(
            np.asarray(samples), state0_b, obs_next_b, env, mean_atol=0.15, var_rtol=0.3
        )

    def test_datagen_gold_smc(self, env, key):
        ptp1, wtp1, truth, obs, forcing, sigma_obs, center = scenario_block(
            env, T, 32768, 2048, jax.random.split(key, 2)
        )
        assert np.allclose(np.asarray(sigma_obs), SIGMA_OBS, atol=1e-6)
        for i in range(2):
            mus, Ps = _kalman_moments(
                env, float(np.asarray(forcing)[i]), np.asarray(center)[i], np.asarray(obs)[i]
            )
            parts = np.asarray(ptp1[i], dtype=np.float64)
            w = np.asarray(wtp1[i], dtype=np.float64)
            for t in range(1, T + 1):
                wm = (w[t][:, None] * parts[t]).sum(axis=0)
                wv = (w[t][:, None] * (parts[t] - wm) ** 2).sum(axis=0)
                assert np.allclose(wm, mus[t - 1], atol=0.2), (i, t)
                assert np.allclose(wv, np.diag(Ps[t - 1]), rtol=0.4, atol=0.05), (i, t)
