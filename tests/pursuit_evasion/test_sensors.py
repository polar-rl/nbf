import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from nbf.domains.pursuit_evasion.env import (
    SENSOR_LAYOUT,
    SENSOR_POS,
    L,
    PursuitEvasionEnv,
    PursuitEvasionObservation,
    PursuitEvasionState,
    sensor_positions,
    sensor_ranges,
)
from nbf.domains.pursuit_evasion.eval.base import make_env_kn, override_sigma
from nbf.domains.pursuit_evasion.policies import make_random_walk_policy
from nbf.eval.base import rollout_episodes
from nbf.filters.likelihood import likelihood_feature_dim, likelihood_features


def _env(k, m, shuffle=True, policy=None):
    return PursuitEvasionEnv(
        num_evaders=k, time_limit=5, policy=policy, shuffle_observations=shuffle, num_sensors=m
    )


def _particles(key, n, k):
    pos = jax.random.uniform(key, (n, k, 2), minval=0.0, maxval=L)
    return PursuitEvasionState(
        positions=pos,
        step_count=jnp.zeros(n, dtype=jnp.int32),
        terminal=jnp.zeros(n, dtype=bool),
        observation_sigma=jnp.full(n, 0.3),
    )


def test_layout_is_nested_and_first_sensor_is_the_original():
    assert jnp.allclose(sensor_positions(1)[0], SENSOR_POS)
    assert jnp.allclose(sensor_positions(4), SENSOR_LAYOUT[:4])
    assert jnp.allclose(sensor_positions(2), sensor_positions(4)[:2])
    with pytest.raises(ValueError):
        sensor_positions(SENSOR_LAYOUT.shape[0] + 1)


def test_single_sensor_matches_original_ranges():
    key = jax.random.PRNGKey(0)
    pos = jax.random.uniform(key, (3, 2), minval=0.0, maxval=L)
    env = _env(3, 1)
    assert jnp.allclose(env.true_ranges(pos), jnp.linalg.norm(pos - SENSOR_POS, axis=-1))
    assert env.reset(key)[1].observation.ranges.shape == (3,)


def test_ranges_are_sensor_major():
    key = jax.random.PRNGKey(1)
    pos = jax.random.uniform(key, (3, 2), minval=0.0, maxval=L)
    env = _env(3, 4)
    flat = env.true_ranges(pos)
    assert flat.shape == (12,)
    per_sensor = sensor_ranges(pos, env.sensors)  # (4, 3)
    assert jnp.allclose(flat.reshape(4, 3), per_sensor)
    for s in range(4):
        assert jnp.allclose(per_sensor[s], jnp.linalg.norm(pos - SENSOR_LAYOUT[s], axis=-1))


def test_step_shuffles_each_sensor_block_independently():
    key = jax.random.PRNGKey(2)
    k, m = 4, 3
    env = _env(k, m, policy=make_random_walk_policy())
    state, _ = env.reset(key)
    ns, ts = env.step(state, 0, jax.random.PRNGKey(3))
    obs = ts.observation.ranges.reshape(m, k)
    truth = sensor_ranges(ns.positions, env.sensors)
    # As sets, each sensor's block is the true block up to noise.
    assert jnp.all(jnp.abs(jnp.sort(obs, axis=1) - jnp.sort(truth, axis=1)) < 5.0 * 0.5)
    # The blocks are not all permuted the same way over many draws.
    perms = []
    for i in range(20):
        _, ts_i = env.step(state, 0, jax.random.PRNGKey(100 + i))
        blocks = ts_i.observation.ranges.reshape(m, k)
        perms.append(jnp.argsort(blocks, axis=1))
    perms = jnp.stack(perms)  # (20, m, k)
    assert bool(jnp.any(perms[:, 0, :] != perms[:, 1, :]))


def test_shuffled_log_prob_sums_over_sensors():
    key = jax.random.PRNGKey(4)
    k, m, n = 3, 4, 16
    parts = _particles(key, n, k)
    env = _env(k, m)
    truth = jax.random.uniform(jax.random.PRNGKey(5), (k, 2), minval=0.0, maxval=L)
    ranges = env.true_ranges(truth) + 0.1
    obs = PursuitEvasionObservation(ranges=ranges, observation_sigma=jnp.asarray(0.3))
    lp = env.observation_log_prob(parts.positions, obs)
    assert lp.shape == (n,)
    # Reference: one single-sensor env per sensor location, ranges block by block.
    total = jnp.zeros(n)
    for s in range(m):
        env_s = eqx.tree_at(lambda e: e.sensors, _env(k, 1), SENSOR_LAYOUT[s : s + 1])
        obs_s = PursuitEvasionObservation(
            ranges=ranges.reshape(m, k)[s], observation_sigma=jnp.asarray(0.3)
        )
        total = total + env_s.observation_log_prob(parts.positions, obs_s)
    assert jnp.allclose(lp, total, atol=1e-4)
    # Weights agree with the softmax of the summed log-prob.
    w, flag = env.observation_weights(parts, None, obs)
    assert jnp.allclose(w, jax.nn.softmax(lp), atol=1e-6) and not bool(flag)


def test_shuffled_log_prob_invariant_to_within_sensor_permutation():
    key = jax.random.PRNGKey(6)
    k, m, n = 3, 2, 8
    parts = _particles(key, n, k)
    env = _env(k, m)
    ranges = jnp.arange(1.0, 1.0 + m * k)
    obs = PursuitEvasionObservation(ranges=ranges, observation_sigma=jnp.asarray(0.4))
    permuted = ranges.reshape(m, k)[:, jnp.array([2, 0, 1])].reshape(-1)
    obs_p = PursuitEvasionObservation(ranges=permuted, observation_sigma=jnp.asarray(0.4))
    assert jnp.allclose(
        env.observation_log_prob(parts.positions, obs),
        env.observation_log_prob(parts.positions, obs_p),
    )
    # Swapping a range across sensors is a different observation.
    swapped = ranges.at[0].set(ranges[k]).at[k].set(ranges[0])
    obs_x = PursuitEvasionObservation(ranges=swapped, observation_sigma=jnp.asarray(0.4))
    assert not jnp.allclose(
        env.observation_log_prob(parts.positions, obs),
        env.observation_log_prob(parts.positions, obs_x),
    )


def test_feature_dims_and_labelled_residuals():
    k, m = 3, 2
    assert likelihood_feature_dim(_env(k, m, shuffle=True)) == 1
    assert likelihood_feature_dim(_env(k, m, shuffle=False)) == m * k
    env = _env(k, m, shuffle=False)
    parts = _particles(jax.random.PRNGKey(7), 5, k)
    ranges = env.true_ranges(parts.positions[0])
    obs = PursuitEvasionObservation(ranges=ranges, observation_sigma=jnp.asarray(0.3))
    feat = likelihood_features(env, parts, obs)
    assert feat.shape == (5, m * k)
    assert jnp.allclose(feat[0], 0.0, atol=1e-5)  # particle 0 is the truth
    feat_s = likelihood_features(_env(k, m, shuffle=True), parts, obs)
    assert feat_s.shape == (5, 1) and float(jnp.max(feat_s)) == 0.0


def test_enkf_operator_sorts_within_sensor_blocks():
    k, m = 3, 2
    env = _env(k, m)
    flat = jax.random.uniform(jax.random.PRNGKey(8), (4, 2 * k), minval=0.0, maxval=L)
    z = env.predicted_observation(flat)
    assert z.shape == (4, m * k)
    blocks = z.reshape(4, m, k)
    assert jnp.all(blocks[:, :, 1:] >= blocks[:, :, :-1])
    obs = PursuitEvasionObservation(ranges=jnp.array([3.0, 1.0, 2.0, 6.0, 4.0, 5.0]))
    assert jnp.allclose(env.observation_vector(obs), jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_rollout_and_override_sigma_shapes():
    k, m, T, B = 3, 4, 5, 2
    env = make_env_kn(k, T, (make_random_walk_policy(),) * k, num_sensors=m)
    episodes = rollout_episodes(env, T, B, jax.random.PRNGKey(9))
    state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = episodes
    assert obs_next_b.ranges.shape == (B, T, m * k)
    assert bool(jnp.all(obs0_b.ranges < 0.0))
    out = override_sigma(env, state0_b, obs0_b, states_next_b, 0.05, jax.random.PRNGKey(10))
    assert out[4].ranges.shape == (B, T, m * k)
    truth = env.true_ranges(states_next_b.positions)
    assert float(jnp.max(jnp.abs(out[4].ranges - truth))) < 0.5
