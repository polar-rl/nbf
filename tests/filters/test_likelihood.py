"""Likelihood features: per-factor sufficient statistics in place of a scalar weight."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nbf.domains.lorenz96.env import OBS_SENTINEL, Lorenz96Env, Lorenz96Observation
from nbf.domains.pursuit_evasion.env import (
    SENSOR_POS,
    PursuitEvasionObservation,
    build_pursuit_evasion_env,
)
from nbf.filters.likelihood import LOG_FEATURE_FLOOR, RESIDUAL_CLIP, log_weight_features

K = 8
OBS = (1, 3, 5, 7)


def _l96():
    return Lorenz96Env(
        K=K,
        time_limit=4,
        sigma_dyn=0.1,
        sigma_obs_min=0.5,
        sigma_obs_max=0.5,
        f_bands=((8.0, 8.0),),
        obs_sites=OBS,
    )


def _particles(env, n, key):
    state, ts = env.reset(key)
    return jax.vmap(lambda k: env.sample_prior(state, k))(jax.random.split(key, n)), state, ts


class TestLorenz96:
    def test_dim_and_layout(self):
        env = _l96()
        assert env.likelihood_feature_dim == K
        particles, state, _ = _particles(env, 6, jax.random.PRNGKey(0))
        y = jnp.arange(len(OBS), dtype=jnp.float32)
        obs = Lorenz96Observation(values=y, sigma_obs=state.sigma_obs)
        z = np.asarray(env.likelihood_features(particles, obs))
        assert z.shape == (6, K)
        unobserved = [k for k in range(K) if k not in OBS]
        assert np.all(z[:, unobserved] == 0.0)
        expected = (np.asarray(y) - np.asarray(particles.x)[:, list(OBS)]) / float(state.sigma_obs)
        assert np.allclose(
            z[:, list(OBS)], np.clip(expected, -RESIDUAL_CLIP, RESIDUAL_CLIP), atol=1e-5
        )
        assert np.any(np.abs(z[:, list(OBS)]) < RESIDUAL_CLIP)

    def test_absent_observation_is_zero(self):
        env = _l96()
        particles, state, ts = _particles(env, 5, jax.random.PRNGKey(1))
        assert float(ts.observation.values[0]) == OBS_SENTINEL
        z = np.asarray(env.likelihood_features(particles, ts.observation))
        assert z.shape == (5, K) and np.all(z == 0.0)

    def test_columns_determine_the_scalar_weight(self):
        env = _l96()
        particles, state, _ = _particles(env, 16, jax.random.PRNGKey(2))
        y = jnp.mean(particles.x[:, list(OBS)], axis=0)
        obs = Lorenz96Observation(values=y, sigma_obs=state.sigma_obs)
        z = env.likelihood_features(particles, obs)
        assert bool(jnp.all(jnp.abs(z) < RESIDUAL_CLIP))
        logw = -0.5 * jnp.sum(z**2, axis=-1)
        w = jax.nn.softmax(logw)
        w_env, _ = env.observation_weights(
            particles, jax.vmap(lambda p: p.observation_from_state())(particles), obs
        )
        assert np.allclose(np.asarray(w), np.asarray(w_env), atol=1e-6)


class TestScalarFeature:
    def test_max_shifted_and_floored(self):
        env = _l96()
        particles, state, _ = _particles(env, 16, jax.random.PRNGKey(3))
        obs = Lorenz96Observation(values=jnp.zeros(len(OBS)), sigma_obs=state.sigma_obs)
        w, _ = env.observation_weights(
            particles, jax.vmap(lambda p: p.observation_from_state())(particles), obs
        )
        f = np.asarray(log_weight_features(w))
        assert f.shape == (16, 1)
        assert np.max(f) == 0.0 and np.min(f) >= LOG_FEATURE_FLOOR
        raw = np.log(np.maximum(np.asarray(w) / np.max(np.asarray(w)), 1e-300))
        assert np.allclose(f[:, 0], np.maximum(raw, LOG_FEATURE_FLOOR), atol=1e-4)

    def test_no_state_observation_means_uniform_weights(self):
        env = _l96()
        particles, state, ts = _particles(env, 4, jax.random.PRNGKey(4))
        w, discarded = env.state_observation_weights(particles, ts.observation)
        assert np.allclose(np.asarray(w), 0.25) and not bool(discarded)


class TestPursuitEvasion:
    def _run(self, k, shuffle):
        env = build_pursuit_evasion_env(num_evaders=k, shuffle_observations=shuffle)
        state, _ = env.reset(jax.random.PRNGKey(5))
        truth = state.observation_from_state()
        obs = PursuitEvasionObservation(
            ranges=truth.ranges + 0.05, observation_sigma=state.observation_sigma
        )
        noise = 0.1 * jax.random.normal(jax.random.PRNGKey(7), (7,) + state.positions.shape)
        particles = jax.tree.map(lambda x: jnp.stack([x] * 7), state)
        particles = eqx.tree_at(lambda s: s.positions, particles, particles.positions + noise)
        return env, particles, obs

    def test_labelled_ranges_factor_per_evader(self):
        env, particles, obs = self._run(3, False)
        assert env.likelihood_feature_dim == 3
        z = env.likelihood_features(particles, obs)
        assert z.shape == (7, 3)
        pred = jnp.linalg.norm(particles.positions - SENSOR_POS, axis=-1)
        expected = (obs.ranges - pred) / obs.observation_sigma
        assert np.allclose(np.asarray(z), np.asarray(expected), atol=1e-5)

    def test_shuffled_ranges_fall_back_to_scalar(self):
        env, particles, obs = self._run(3, True)
        assert env.likelihood_feature_dim == 1
        z = np.asarray(env.likelihood_features(particles, obs))
        assert z.shape == (7, 1) and np.max(z) == 0.0 and np.min(z) >= LOG_FEATURE_FLOOR

    def test_single_evader_is_one_factor(self):
        env, particles, obs = self._run(1, False)
        assert env.likelihood_feature_dim == 1
        assert env.likelihood_features(particles, obs).shape == (7, 1)

    def test_absent_observation_is_zero(self):
        env, particles, _ = self._run(2, False)
        _, ts = env.reset(jax.random.PRNGKey(8))
        z = np.asarray(env.likelihood_features(particles, ts.observation))
        assert z.shape == (7, 2) and np.all(z == 0.0)
