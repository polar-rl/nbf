import jax
import numpy as np
import pytest

from nbf.domains.lorenz96.env import Lorenz96Env
from nbf.filters.particle import ParticleFilter, RegularizedParticleFilter
from nbf.filters.resample_move import ResampleMoveParticleFilter

K = 8
N = 512


@pytest.fixture
def env():
    return Lorenz96Env(
        K=K,
        time_limit=5,
        sigma_dyn=0.3,
        sigma_obs_min=0.1,
        sigma_obs_max=0.1,
        f_bands=((7.9, 8.1),),
        spinup_time=2.0,
    )


def _one_update(env, filt, key):
    reset_key, filt_key, step_key, upd_key = jax.random.split(key, 4)
    state, ts = env.reset(reset_key)
    pset = filt.reset(env, state, ts.observation, filt_key)
    state, ts = env.step(state, 0, step_key)
    pset, neff, _, _ = filt.update(env, pset, None, None, 0, ts.observation, upd_key)
    return pset, neff


def _unique_fraction(pset):
    x = np.asarray(pset.particles.x)
    return np.unique(x, axis=0).shape[0] / x.shape[0]


class TestResampleMove:
    def test_exact_move_removes_duplicates(self, env, key):
        filt = ResampleMoveParticleFilter(num_particles=N, resample_factor=2.0)
        pset, _ = _one_update(env, filt, key)
        assert _unique_fraction(pset) == 1.0
        assert np.allclose(np.asarray(pset.weights), 1.0 / N)

    def test_plain_pf_duplicates_under_same_collapse(self, env, key):
        filt = ParticleFilter(num_particles=N, resample_factor=2.0)
        pset, _ = _one_update(env, filt, key)
        assert _unique_fraction(pset) < 1.0

    def test_mh_move_matches_exact_move_moments(self, env, key):
        exact = ResampleMoveParticleFilter(num_particles=4096, resample_factor=2.0)
        mh = ResampleMoveParticleFilter(
            num_particles=4096, resample_factor=2.0, num_moves=50, step_size=0.15
        )
        pset_e, _ = _one_update(env, exact, key)
        pset_m, _ = _one_update(env, mh, key)
        xe = np.asarray(pset_e.particles.x)
        xm = np.asarray(pset_m.particles.x)
        assert np.allclose(xe.mean(axis=0), xm.mean(axis=0), atol=0.15)
        assert np.allclose(xe.std(axis=0), xm.std(axis=0), atol=0.15)

    def test_no_observation_skips_move(self, env, key):
        filt = ResampleMoveParticleFilter(num_particles=N, resample_factor=2.0)
        reset_key, filt_key, upd_key = jax.random.split(key, 3)
        state, ts = env.reset(reset_key)
        pset = filt.reset(env, state, ts.observation, filt_key)
        pset, _, _, _ = filt.update(env, pset, None, None, 0, ts.observation, upd_key)
        assert bool(np.isfinite(np.asarray(pset.particles.x)).all())

    def test_sigma_dyn_zero_raises_for_exact_move(self, key):
        env0 = Lorenz96Env(
            K=K,
            time_limit=5,
            sigma_dyn=0.0,
            sigma_obs_min=0.1,
            sigma_obs_max=0.1,
            f_bands=((7.9, 8.1),),
            spinup_time=2.0,
        )
        filt = ResampleMoveParticleFilter(num_particles=N, resample_factor=2.0)
        with pytest.raises(ValueError, match='sigma_dyn'):
            _one_update(env0, filt, key)


class TestRegularizedPF:
    def test_jitter_removes_duplicates(self, env, key):
        filt = RegularizedParticleFilter(num_particles=N, resample_factor=2.0)
        pset, _ = _one_update(env, filt, key)
        assert _unique_fraction(pset) == 1.0

    def test_no_resample_no_jitter(self, env, key):
        pf = ParticleFilter(num_particles=N, resample_factor=0.0)
        reg = RegularizedParticleFilter(num_particles=N, resample_factor=0.0)
        pset_pf, _ = _one_update(env, pf, key)
        pset_reg, _ = _one_update(env, reg, key)
        assert np.allclose(np.asarray(pset_pf.particles.x), np.asarray(pset_reg.particles.x))
