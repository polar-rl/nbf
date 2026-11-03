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
    assert _env(k, m, shuffle=True).likelihood_feature_dim == 1
    assert _env(k, m, shuffle=False).likelihood_feature_dim == m * k
    env = _env(k, m, shuffle=False)
    parts = _particles(jax.random.PRNGKey(7), 5, k)
    ranges = env.true_ranges(parts.positions[0])
    obs = PursuitEvasionObservation(ranges=ranges, observation_sigma=jnp.asarray(0.3))
    feat = env.likelihood_features(parts, obs)
    assert feat.shape == (5, m * k)
    assert jnp.allclose(feat[0], 0.0, atol=1e-5)  # particle 0 is the truth
    feat_s = _env(k, m, shuffle=True).likelihood_features(parts, obs)
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


def test_obs_interval_emits_sentinel_between_observations():
    k, m, D, T = 3, 2, 5, 20
    env = PursuitEvasionEnv(
        num_evaders=k,
        time_limit=T,
        policy=make_random_walk_policy(),
        shuffle_observations=True,
        num_sensors=m,
        obs_interval=D,
    )
    assert jnp.array_equal(env.observation_mask(T), (jnp.arange(1, T + 1) % D) == 0)
    state, _ = env.reset(jax.random.PRNGKey(11))
    seen = []
    for t in range(T):
        state, ts = env.step(state, 0, jax.random.PRNGKey(20 + t))
        seen.append(bool(env.observation_present(ts.observation)))
        if not seen[-1]:
            assert bool(jnp.all(ts.observation.ranges == -1.0))
    assert seen == [((t + 1) % D) == 0 for t in range(T)]
    # Unobserved steps weight uniformly and carry a zero feature.
    parts = _particles(jax.random.PRNGKey(12), 8, k)
    sentinel = PursuitEvasionObservation(
        ranges=jnp.full(m * k, -1.0), observation_sigma=jnp.asarray(0.3)
    )
    w, _ = env.observation_weights(parts, None, sentinel)
    assert jnp.allclose(w, 1.0 / 8)
    assert float(jnp.abs(env.likelihood_features(parts, sentinel)).max()) == 0.0
    with pytest.raises(ValueError):
        PursuitEvasionEnv(num_evaders=k, obs_interval=0)


def test_override_sigma_keeps_unobserved_steps():
    k, m, D, T, B = 3, 2, 4, 8, 2
    env = make_env_kn(k, T, (make_random_walk_policy(),) * k, num_sensors=m, obs_interval=D)
    state0_b, obs0_b, states_next_b, _, obs_next_b = rollout_episodes(
        env, T, B, jax.random.PRNGKey(13)
    )
    out = override_sigma(env, state0_b, obs0_b, states_next_b, 0.05, jax.random.PRNGKey(14))
    mask = env.observation_mask(T)
    assert bool(jnp.all(out[4].ranges[:, ~mask] == -1.0)) and bool(
        jnp.all(out[4].ranges[:, mask] >= 0.0)
    )
    assert bool(jnp.all((obs_next_b.ranges[:, :, 0] < 0) == ~mask[None, :]))


def test_noise_range_is_configurable():
    env = PursuitEvasionEnv(num_evaders=2, sigma_obs_min=0.01, sigma_obs_max=0.05)
    sig = jnp.stack([env.reset(jax.random.PRNGKey(i))[0].observation_sigma for i in range(64)])
    assert float(sig.min()) >= 0.01 and float(sig.max()) <= 0.05 and float(sig.std()) > 0.0
    default = PursuitEvasionEnv(num_evaders=2)
    sig_d = jnp.stack(
        [default.reset(jax.random.PRNGKey(i))[0].observation_sigma for i in range(64)]
    )
    assert float(sig_d.min()) >= 0.1 and float(sig_d.max()) <= 0.5
    with pytest.raises(ValueError):
        PursuitEvasionEnv(num_evaders=2, sigma_obs_min=0.1, sigma_obs_max=0.05)


def test_feature_modes_shuffled():
    k, n = 3, 6
    parts = _particles(jax.random.PRNGKey(21), n, k)
    ranges = PursuitEvasionEnv(num_evaders=k, shuffle_observations=True).true_ranges(
        parts.positions[0]
    )
    obs = PursuitEvasionObservation(ranges=ranges, observation_sigma=jnp.asarray(0.01))
    envs = {
        m: PursuitEvasionEnv(num_evaders=k, shuffle_observations=True, feature_mode=m)
        for m in ('nats', 'range', 'both')
    }
    f = {m: e.likelihood_features(parts, obs) for m, e in envs.items()}
    assert f['nats'].shape == (n, 1) and f['range'].shape == (n, 1) and f['both'].shape == (n, 2)
    assert envs['both'].likelihood_feature_dim == 2
    assert jnp.allclose(f['both'][:, :1], f['nats']) and jnp.allclose(f['both'][:, 1:], f['range'])
    # Particle 0 is the truth: best particle sits at 0 in both units.
    assert float(f['nats'][0, 0]) == 0.0 and float(f['range'][0, 0]) == 0.0
    # At sigma 0.01 the nats feature floors every other particle; the range feature stays graded.
    assert bool(jnp.all(f['nats'][1:, 0] == -30.0))
    rng = f['range'][1:, 0]
    assert bool(jnp.all(rng < 0.0)) and float(jnp.std(rng)) > 0.0 and float(rng.min()) >= -20.0
    # Range feature is sigma-invariant: same positions at sigma 0.05 give the same range block.
    obs5 = PursuitEvasionObservation(ranges=ranges, observation_sigma=jnp.asarray(0.05))
    assert jnp.allclose(envs['range'].likelihood_features(parts, obs5), f['range'], atol=1e-4)
    with pytest.raises(ValueError):
        PursuitEvasionEnv(num_evaders=k, feature_mode='bogus')


def test_feature_modes_labelled():
    k, n = 2, 4
    env_b = PursuitEvasionEnv(num_evaders=k, shuffle_observations=False, feature_mode='both')
    parts = _particles(jax.random.PRNGKey(22), n, k)
    ranges = env_b.true_ranges(parts.positions[0]) + 0.1
    obs = PursuitEvasionObservation(ranges=ranges, observation_sigma=jnp.asarray(0.05))
    feat = env_b.likelihood_features(parts, obs)
    assert feat.shape == (n, 2 * k) and env_b.likelihood_feature_dim == 2 * k
    assert jnp.allclose(feat[0, :k], 2.0, atol=1e-4) and jnp.allclose(feat[0, k:], 0.1, atol=1e-4)
