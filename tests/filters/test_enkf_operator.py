import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nbf.domains.lorenz96.env import Lorenz96Env
from nbf.domains.pursuit_evasion.env import (
    SENSOR_POS,
    L,
    PursuitEvasionEnv,
    PursuitEvasionObservation,
)
from nbf.domains.pursuit_evasion.eval.dispatch import (
    PE_EXTRA_DENSITY_SAMPLERS,
    PE_EXTRA_SAMPLERS,
    make_pe_enkf,
)
from nbf.domains.pursuit_evasion.policies import make_random_walk_policy
from nbf.filters.enkf import EnsembleKalmanFilter, LocalEnsembleTransformKalmanFilter


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def _pe_env(k: int, shuffle: bool = True, time_limit: int = 5) -> PursuitEvasionEnv:
    return PursuitEvasionEnv(
        num_evaders=k,
        time_limit=time_limit,
        policy=(make_random_walk_policy(),) * k,
        shuffle_observations=shuffle,
    )


class TestOperatorHooks:
    def test_sorted_ranges_are_label_invariant(self, key):
        env = _pe_env(3)
        pos = jax.random.uniform(key, (8, 3, 2), minval=0.0, maxval=L)
        flat = pos.reshape(8, -1)
        z = env.predicted_observation(flat)
        assert z.shape == (8, 3)
        assert jnp.all(z[:, :-1] <= z[:, 1:])
        perm = pos[:, ::-1, :].reshape(8, -1)
        assert jnp.allclose(env.predicted_observation(perm), z)
        ranges = jnp.linalg.norm(pos - SENSOR_POS, axis=-1)
        assert jnp.allclose(z, jnp.sort(ranges, axis=-1))

    def test_labelled_ranges_keep_order(self, key):
        env = _pe_env(2, shuffle=False)
        pos = jax.random.uniform(key, (4, 2, 2), minval=0.0, maxval=L)
        z = env.predicted_observation(pos.reshape(4, -1))
        assert jnp.allclose(z, jnp.linalg.norm(pos - SENSOR_POS, axis=-1))
        obs = PursuitEvasionObservation(ranges=jnp.array([3.0, 1.0]), observation_sigma=0.2)
        assert jnp.allclose(env.observation_vector(obs), obs.ranges)
        assert float(env.observation_noise_std(obs)) == pytest.approx(0.2)

    def test_observation_vector_sorted_under_shuffle(self):
        env = _pe_env(3)
        obs = PursuitEvasionObservation(ranges=jnp.array([3.0, 1.0, 2.0]), observation_sigma=0.3)
        assert jnp.allclose(env.observation_vector(obs), jnp.array([1.0, 2.0, 3.0]))


class _LinearOperatorEnv(eqx.Module):
    """Wraps a Lorenz-96 env and exposes its linear observation through the hooks,
    without `obs_idx`, so the two analysis paths can be compared on one problem."""

    inner: Lorenz96Env

    def predicted_observation(self, flat):
        return flat[:, self.inner.obs_idx]

    def observation_vector(self, observation):
        return observation.values

    def observation_noise_std(self, observation):
        return observation.sigma_obs


class TestOperatorPathMatchesLinearPath:
    def test_same_analysis_without_localization(self, key):
        env = Lorenz96Env(K=8, time_limit=3, sigma_obs_min=0.7, sigma_obs_max=0.7)
        state, ts = env.reset(key)
        state, ts = env.step(state, 0, key)
        enkf = EnsembleKalmanFilter(num_members=64, inflation=1.02, loc_radius=1e6)
        X = state.x[None] + jax.random.normal(key, (64, 8))
        analysis_key = jax.random.PRNGKey(3)
        linear = enkf._analysis(X, ts.observation, env.obs_idx, analysis_key)
        operator = enkf._analysis_operator(_LinearOperatorEnv(env), X, ts.observation, analysis_key)
        assert np.allclose(np.asarray(linear), np.asarray(operator), atol=1e-4)


class TestPursuitEvasionEnKF:
    def test_update_keeps_members_valid_and_uniform(self, key):
        env = _pe_env(3)
        enkf = make_pe_enkf(32)
        state, ts = env.reset(key)
        pset = enkf.reset(env, state, ts.observation, key)
        assert pset.particles.positions.shape == (32, 3, 2)
        state, ts = env.step(state, 0, key)
        new, neff, _, _ = enkf.update(env, pset, None, None, 0, ts.observation, key)
        pos = np.asarray(new.particles.positions)
        assert pos.shape == (32, 3, 2)
        assert np.all(pos >= 0.0) and np.all(pos <= L)
        assert np.allclose(new.weights, 1.0 / 32)
        assert float(neff) == pytest.approx(32.0)

    def test_analysis_pulls_predicted_ranges_toward_observation(self, key):
        env = _pe_env(1, shuffle=False)
        enkf = EnsembleKalmanFilter(num_members=256, inflation=1.0)
        truth = jnp.array([[2.0, 8.0]])
        members = truth[None] + 0.7 * jax.random.normal(key, (256, 1, 2))
        sharp = PursuitEvasionObservation(
            ranges=jnp.linalg.norm(truth - SENSOR_POS, axis=-1),
            observation_sigma=jnp.asarray(0.05),
        )
        before = env.predicted_observation(members.reshape(256, -1))
        X = enkf._analyse(env, members, sharp, key)
        after = env.predicted_observation(X.reshape(256, -1))
        err_before = float(jnp.mean(jnp.abs(before - sharp.ranges)))
        err_after = float(jnp.mean(jnp.abs(after - sharp.ranges)))
        assert err_after < 0.5 * err_before

    def test_letkf_refuses_operator_env(self, key):
        env = _pe_env(2)
        letkf = LocalEnsembleTransformKalmanFilter(num_members=8)
        state, ts = env.reset(key)
        pset = letkf.reset(env, state, ts.observation, key)
        state, ts = env.step(state, 0, key)
        with pytest.raises(ValueError, match='obs_idx'):
            letkf.update(env, pset, None, None, 0, ts.observation, key)

    def test_registered_samplers(self):
        assert 'enkf' in PE_EXTRA_SAMPLERS
        assert 'enkf' in PE_EXTRA_DENSITY_SAMPLERS


class TestPursuitEvasionEvalPath:
    def test_samples_and_density_stream(self, key):
        from nbf.domains.pursuit_evasion.eval.dispatch import (
            compute_method_samples,
            compute_method_samples_and_density,
        )
        from nbf.eval.base import rollout_episodes

        env = _pe_env(3, time_limit=4)
        state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = rollout_episodes(env, 4, 3, key)
        common = dict(
            env=env,
            embedding_model=None,
            state0_b=state0_b,
            obs0_b=obs0_b,
            states_next_b=states_next_b,
            obs_prev_b=obs_prev_b,
            obs_next_b=obs_next_b,
            num_ref_samples=16,
            key=key,
        )
        samples = compute_method_samples('enkf_12', **common)
        assert samples.shape == (3, 4, 16, 3, 2)
        samples, metrics = compute_method_samples_and_density('enkf_12', **common)
        assert samples.shape == (3, 4, 16, 3, 2)
        nll = np.asarray(metrics['nll'])
        assert nll.shape == (3, 4)
        assert np.all(np.isfinite(nll))

    def test_score_methods_with_relabelling(self, key):
        from nbf.domains.pursuit_evasion.eval.dispatch import (
            compute_method_samples,
            compute_method_samples_and_density,
        )
        from nbf.domains.pursuit_evasion.run_grid import DENSITY_KINDS
        from nbf.eval.base import rollout_episodes
        from nbf.eval.grid import score_methods

        env = _pe_env(2, time_limit=3)
        episodes = rollout_episodes(env, 3, 2, key)
        out = score_methods(
            ['pf_16', 'enkf_8'],
            tag='smoke',
            key=key,
            env=env,
            episodes=episodes,
            truth=episodes[2].positions,
            sampling_model=None,
            density_model=None,
            density_kinds=DENSITY_KINDS,
            compute_samples=compute_method_samples,
            compute_samples_and_density=compute_method_samples_and_density,
            extra_samplers=PE_EXTRA_SAMPLERS,
            num_ref_samples=8,
            chunk_size=0,
            timing_repeats=0,
            t_min=0,
            relabel_perms=env.perm_indices,
        )
        assert set(out) == {'pf_16', 'enkf_8'}
        assert 'nll' in out['enkf_8'] and 'nll' not in out['pf_16']
        for spec in out:
            for name in ('rmse', 'crps', 'cov50', 'cov90', 'spread'):
                assert out[spec][name].shape == (2, 3), (spec, name)
                assert np.all(np.isfinite(out[spec][name]))
