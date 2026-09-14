import jax
import jax.numpy as jnp
import pytest
from jumanji.types import StepType

from nbf.domains.pursuit_evasion.env import (
    DEFAULT_OBSTACLES,
    SENSOR_POS,
    SIGMA_OBS_MAX,
    SPEED,
    L,
    PursuitEvasionEnv,
    PursuitEvasionGenerator,
    PursuitEvasionState,
    clip_to_valid,
    propagate_deterministic,
    transition_single,
)
from nbf.domains.pursuit_evasion.policies import make_goal_policy, make_random_walk_policy


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def env():
    return PursuitEvasionEnv(num_evaders=1, time_limit=5)


def _inside_any_obstacle(pos, obstacles):
    for i in range(obstacles.shape[0]):
        o = obstacles[i]
        if o[0] < pos[0] < o[2] and o[1] < pos[1] < o[3]:
            return True
    return False


class TestClipToValid:
    def test_clips_below_zero(self):
        out = clip_to_valid(jnp.array([-1.5, 3.0]), DEFAULT_OBSTACLES)
        assert out[0] == pytest.approx(1.5)
        assert out[1] == pytest.approx(3.0)

    def test_clips_above_L(self):
        out = clip_to_valid(jnp.array([L + 1.0, L + 0.25]), DEFAULT_OBSTACLES)
        assert out[0] == pytest.approx(L - 1.0)
        assert out[1] == pytest.approx(L - 0.25)

    def test_keeps_interior_point(self):
        p = jnp.array([1.0, 1.0])
        out = clip_to_valid(p, DEFAULT_OBSTACLES)
        assert jnp.allclose(out, p)

    def test_pushes_point_out_of_obstacle(self):
        # interior of the central 4-6 square
        p = jnp.array([5.0, 5.1])
        out = clip_to_valid(p, DEFAULT_OBSTACLES)
        assert not _inside_any_obstacle(out, DEFAULT_OBSTACLES)


class TestTransitionSingle:
    def test_output_shape_and_in_bounds(self, key):
        policy = make_random_walk_policy()
        for sub in jax.random.split(key, 16):
            start = jax.random.uniform(sub, (2,), minval=0.5, maxval=L - 0.5)
            nxt = transition_single(start, policy, DEFAULT_OBSTACLES, sub)
            assert nxt.shape == (2,)
            assert 0.0 <= float(nxt[0]) <= L
            assert 0.0 <= float(nxt[1]) <= L
            assert not _inside_any_obstacle(nxt, DEFAULT_OBSTACLES)

    def test_deterministic_step_moves_toward_goal(self):
        policy = make_goal_policy(jnp.array([9.0, 9.0]), concentration=100.0)
        start = jnp.array([1.0, 1.0])
        nxt = propagate_deterministic(start, policy, DEFAULT_OBSTACLES)
        d_before = float(jnp.linalg.norm(jnp.array([9.0, 9.0]) - start))
        d_after = float(jnp.linalg.norm(jnp.array([9.0, 9.0]) - nxt))
        assert d_before - d_after == pytest.approx(SPEED, abs=1e-4)


class TestState:
    def test_flat_features_roundtrip(self, key):
        positions = jax.random.uniform(key, (1, 2), minval=0.0, maxval=L)
        s = PursuitEvasionState(positions=positions, step_count=3, terminal=False)
        flat = s.to_flat_features()
        assert flat.shape == (2,)
        s2 = s.from_flat_features(flat)
        assert jnp.allclose(s2.positions, positions)
        assert s2.step_count == s.step_count
        assert s2.terminal == s.terminal

    def test_observation_ranges_match_distance(self, key):
        positions = jax.random.uniform(key, (1, 2), minval=0.0, maxval=L)
        s = PursuitEvasionState(positions=positions, step_count=0, terminal=False)
        obs = s.observation_from_state()
        expected = jnp.linalg.norm(positions - SENSOR_POS, axis=-1)
        assert jnp.allclose(obs.ranges, expected)


class TestGenerator:
    def test_positions_in_bounds_and_not_in_obstacles(self, key):
        gen = PursuitEvasionGenerator(num_evaders=1, obstacles=DEFAULT_OBSTACLES)
        for sub in jax.random.split(key, 32):
            state = gen(sub)
            assert state.positions.shape == (1, 2)
            assert state.step_count == 0
            assert bool(state.terminal) is False
            for p in state.positions:
                assert 0.0 <= float(p[0]) <= L
                assert 0.0 <= float(p[1]) <= L
                assert not _inside_any_obstacle(p, DEFAULT_OBSTACLES)


class TestEnvReset:
    def test_initial_obs_is_sentinel(self, env, key):
        _, ts = env.reset(key)
        assert ts.observation.ranges.shape == (env.num_evaders,)
        assert jnp.all(ts.observation.ranges == -1.0)
        assert ts.step_type == StepType.FIRST


class TestEnvStep:
    def test_step_advances_position_by_about_speed(self, key):
        policy = make_goal_policy(jnp.array([9.0, 9.0]), concentration=100.0)
        env_p = PursuitEvasionEnv(num_evaders=1, time_limit=5, policy=policy)
        start = jnp.array([[1.0, 1.0]])
        state = PursuitEvasionState(positions=start, step_count=0, terminal=False)
        deltas = []
        for sub in jax.random.split(key, 64):
            ns, _ = env_p.step(state, 0, sub)
            deltas.append(float(jnp.linalg.norm(ns.positions - start)))
        mean_step = sum(deltas) / len(deltas)
        assert SPEED * 0.5 < mean_step < SPEED * 2.0

    def test_step_observation_close_to_true_range(self, key):
        policy = make_random_walk_policy()
        env_p = PursuitEvasionEnv(num_evaders=1, time_limit=5, policy=policy)
        state, _ = env_p.reset(key)
        ns, ts = env_p.step(state, 0, key)
        true_r = float(jnp.linalg.norm(ns.positions[0] - SENSOR_POS))
        assert abs(float(ts.observation.ranges[0]) - true_r) < 5.0 * SIGMA_OBS_MAX

    def test_terminates_at_time_limit(self, key):
        policy = make_random_walk_policy()
        env_p = PursuitEvasionEnv(num_evaders=1, time_limit=3, policy=policy)
        state, _ = env_p.reset(key)
        ts = None
        for sub in jax.random.split(key, env_p.time_limit):
            state, ts = env_p.step(state, 0, sub)
        assert int(state.step_count) == env_p.time_limit
        assert bool(state.terminal) is True
        assert ts.step_type == StepType.LAST


class TestObservationWeights:
    def test_uniform_when_no_observation(self, env, key):
        n = 16
        keys = jax.random.split(key, n)
        states, timesteps = jax.vmap(env.reset)(keys)
        gold = jax.tree.map(lambda x: x[0], timesteps.observation)
        w, _ = env.observation_weights(states, timesteps.observation, gold)
        assert w.shape == (n,)
        assert jnp.allclose(w, jnp.ones(n) / n)

    def test_softmax_when_observation_present(self, key):
        policy = make_random_walk_policy()
        env_p = PursuitEvasionEnv(num_evaders=1, time_limit=5, policy=policy)
        n = 32
        keys = jax.random.split(key, n)
        states, _ = jax.vmap(env_p.reset)(keys)
        step_keys = jax.random.split(key, n)
        next_states, next_ts = jax.vmap(lambda s, k: env_p.step(s, 0, k))(states, step_keys)

        gold = jax.tree.map(lambda x: x[0], next_ts.observation)
        w, _ = env_p.observation_weights(next_states, next_ts.observation, gold)
        assert w.shape == (n,)
        assert jnp.all(w >= 0)
        assert float(jnp.sum(w)) == pytest.approx(1.0, rel=1e-5)
        assert float(jnp.std(w)) > 0.0


class TestMultiEvader:
    def test_repulsion_pushes_close_pair_apart(self, key):
        pols = (make_random_walk_policy(),) * 2
        env_no = PursuitEvasionEnv(
            num_evaders=2,
            time_limit=5,
            policy=pols,
            interaction_strength=0.0,
            interaction_radius=1.5,
        )
        env_yes = PursuitEvasionEnv(
            num_evaders=2,
            time_limit=5,
            policy=pols,
            interaction_strength=2.0,
            interaction_radius=2.0,
        )
        positions = jnp.array([[5.0, 5.0], [5.2, 5.0]])
        state = PursuitEvasionState(positions=positions, step_count=0, terminal=False)
        d_no, d_yes = [], []
        for sub in jax.random.split(key, 64):
            ns_no, _ = env_no.step(state, 0, sub)
            ns_yes, _ = env_yes.step(state, 0, sub)
            d_no.append(float(jnp.linalg.norm(ns_no.positions[0] - ns_no.positions[1])))
            d_yes.append(float(jnp.linalg.norm(ns_yes.positions[0] - ns_yes.positions[1])))
        assert sum(d_yes) / len(d_yes) > sum(d_no) / len(d_no) + 0.05

    def test_shuffled_obs_ranges_match_set(self, key):
        pols = (make_random_walk_policy(),) * 3
        env_p = PursuitEvasionEnv(
            num_evaders=3,
            time_limit=5,
            policy=pols,
            shuffle_observations=True,
        )
        state, _ = env_p.reset(key)
        ns, ts = env_p.step(state, 0, key)
        true_ranges = jnp.linalg.norm(ns.positions - SENSOR_POS, axis=-1)
        sorted_obs = jnp.sort(ts.observation.ranges)
        sorted_true = jnp.sort(true_ranges)
        max_dev = float(jnp.max(jnp.abs(sorted_obs - sorted_true)))
        assert max_dev < 5.0 * SIGMA_OBS_MAX

    def test_observation_weights_permutation_invariant(self, key):
        pols = (make_random_walk_policy(),) * 2
        env_p = PursuitEvasionEnv(
            num_evaders=2,
            time_limit=5,
            policy=pols,
            shuffle_observations=True,
        )
        n = 16
        keys = jax.random.split(key, n)
        particles = jax.vmap(env_p.generator)(keys)
        from nbf.domains.pursuit_evasion.env import PursuitEvasionObservation

        obs = PursuitEvasionObservation(ranges=jnp.array([3.0, 7.0]))
        obs_perm = PursuitEvasionObservation(ranges=jnp.array([7.0, 3.0]))
        w, _ = env_p.observation_weights(particles, obs, obs)
        w_perm, _ = env_p.observation_weights(particles, obs_perm, obs_perm)
        assert jnp.allclose(w, w_perm, atol=1e-5)


def test_normalize_log_jacobian_matches_change_of_variables():
    from nbf.domains.pursuit_evasion.env import L

    env = PursuitEvasionEnv(num_evaders=2, time_limit=5)
    x = jnp.array([[[1.0, 2.0], [7.5, 9.0]]])  # (1, k, 2) raw positions
    z = env.normalize_flat(x)[0]
    logp_z = -0.5 * jnp.sum(z**2) - 0.5 * 4 * jnp.log(2 * jnp.pi)
    logp_x = logp_z + env.normalize_log_jacobian()
    expected = jax.scipy.stats.norm.logpdf(x.reshape(-1), loc=0.5 * L, scale=0.5 * L).sum()
    assert jnp.allclose(logp_x, expected, atol=1e-5)
