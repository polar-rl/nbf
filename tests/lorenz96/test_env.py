import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jumanji.types import StepType
from scipy.integrate import solve_ivp

from nbf.domains.lorenz96.env import (
    OBS_SENTINEL,
    Lorenz96Env,
    Lorenz96Observation,
    conditional_gaussian_redraw,
    l96_advection,
    l96_tendency,
    rk4_window,
    rw_mh_redraw,
)
from nbf.filters.particle import ParticleFilter, ParticleSet, RegularizedParticleFilter


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def env():
    return Lorenz96Env(K=8, time_limit=5)


class TestDynamics:
    def test_tendency_matches_hand_loop_k5(self):
        x = np.array([1.0, 2.0, -1.0, 0.5, 3.0])
        F = 8.0
        expected = np.zeros(5)
        for k in range(5):
            expected[k] = (x[(k + 1) % 5] - x[(k - 2) % 5]) * x[(k - 1) % 5] - x[k] + F
        got = l96_tendency(jnp.asarray(x), F)
        assert np.allclose(got, expected, atol=1e-5)

    def test_advection_conserves_energy(self, key):
        x = 3.0 * jax.random.normal(key, (20,))
        assert float(jnp.sum(x * l96_advection(x))) == pytest.approx(0.0, abs=1e-2)

    def test_rk4_window_matches_solve_ivp(self, key):
        x0 = np.asarray(8.0 + 0.5 * jax.random.normal(key, (8,)), dtype=np.float64)
        F = 8.0

        def f(t, x):
            return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

        sol = solve_ivp(f, (0.0, 0.05), x0, rtol=1e-10, atol=1e-12)
        got = rk4_window(jnp.asarray(x0, dtype=jnp.float32), F, 0.01, 5)
        assert np.allclose(got, sol.y[:, -1], atol=1e-4)

    def test_rk4_substep_self_consistency(self, key):
        x0 = 8.0 + 0.5 * jax.random.normal(key, (8,))
        coarse = rk4_window(x0, 8.0, 0.01, 5)
        fine = rk4_window(x0, 8.0, 0.001, 50)
        assert np.allclose(coarse, fine, atol=1e-4)

    def test_climatology_f8_k20(self, key):
        x0 = 8.0 + 0.5 * jax.random.normal(key, (20,))
        x = jax.lax.fori_loop(0, 200, lambda _, xi: rk4_window(xi, 8.0, 0.01, 5), x0)

        def collect(xi, _):
            xi = rk4_window(xi, 8.0, 0.01, 5)
            return xi, xi

        _, xs = jax.lax.scan(collect, x, None, length=2000)
        mean = float(jnp.mean(xs))
        std = float(jnp.std(xs))
        assert 1.5 < mean < 3.1
        assert 2.8 < std < 4.4

    def test_linear_mode_contracts_to_forcing(self):
        x0 = jnp.array([10.0, -4.0, 3.0, 0.0])
        x = x0
        for _ in range(200):
            x = rk4_window(x, 8.0, 0.01, 5, linear=True)
        assert np.allclose(x, 8.0, atol=1e-3)


class TestStateObservation:
    def test_flat_features_roundtrip_with_normalization(self, key):
        env = Lorenz96Env(K=8, time_limit=5, norm_mean=2.3, norm_std=3.6)
        state = env.generator(key)
        flat = state.to_flat_features()
        assert flat.shape == (8,)
        assert np.allclose(flat, (state.x - 2.3) / 3.6, atol=1e-6)
        s2 = state.from_flat_features(flat)
        assert np.allclose(s2.x, state.x, atol=1e-5)
        assert np.allclose(s2.center, state.center)
        assert float(s2.forcing) == pytest.approx(float(state.forcing))

    def test_observation_selects_odd_sites(self, key):
        env = Lorenz96Env(K=8, time_limit=5)
        state = env.generator(key)
        obs = state.observation_from_state()
        assert np.allclose(obs.values, state.x[1::2])
        assert float(obs.sigma_obs) == pytest.approx(float(state.sigma_obs))

    def test_reset_obs_is_sentinel(self, env, key):
        state, ts = env.reset(key)
        assert ts.observation.values.shape == (4,)
        assert np.all(np.asarray(ts.observation.values) == OBS_SENTINEL)
        assert not bool(env.observation_present(ts.observation))
        assert ts.step_type == StepType.FIRST


class TestGenerator:
    def test_fields_and_attractor(self, key):
        env = Lorenz96Env(K=8, time_limit=5)
        for sub in jax.random.split(key, 8):
            state = env.generator(sub)
            F = float(state.forcing)
            assert (5.0 <= F <= 7.0) or (9.0 <= F <= 11.0)
            assert 0.5 <= float(state.sigma_obs) <= 1.5
            assert int(state.step_count) == 0
            assert not bool(state.terminal)
            assert float(jnp.max(jnp.abs(state.x))) < 25.0
            assert float(jnp.std(state.x)) > 0.5
            delta = state.center - state.x
            assert float(jnp.linalg.norm(delta)) < 6.0 * np.sqrt(8)


class TestEnvStep:
    def test_step_counts_and_termination(self, env, key):
        state, _ = env.reset(key)
        ts = None
        for sub in jax.random.split(key, env.time_limit):
            state, ts = env.step(state, 0, sub)
        assert int(state.step_count) == env.time_limit
        assert bool(state.terminal)
        assert ts.step_type == StepType.LAST

    def test_step_observation_close_to_true_state(self, env, key):
        state, _ = env.reset(key)
        next_state, ts = env.step(state, 0, key)
        true_vals = np.asarray(next_state.x[1::2])
        assert np.all(np.abs(np.asarray(ts.observation.values) - true_vals) < 5.0 * 1.5)

    def test_step_noise_magnitude(self, key):
        env = Lorenz96Env(K=8, time_limit=5, sigma_dyn=0.25)
        state, _ = env.reset(key)
        diffs = []
        det = rk4_window(state.x, state.forcing, env.dt, env.substeps)
        for sub in jax.random.split(key, 64):
            ns, _ = env.step(state, 0, sub)
            diffs.append(np.asarray(ns.x - det))
        std = np.std(np.stack(diffs))
        assert 0.2 < std < 0.3


class TestObservationWeights:
    def test_uniform_when_sentinel(self, env, key):
        state, ts = env.reset(key)
        particles = jax.vmap(lambda k: env.sample_prior(state, k))(jax.random.split(key, 16))
        w, _ = env.observation_weights(particles, None, ts.observation)
        assert np.allclose(w, np.ones(16) / 16)

    def test_nonfinite_particle_gets_zero_weight(self, env, key):
        state, _ = env.reset(key)
        particles = jax.vmap(lambda k: env.sample_prior(state, k))(jax.random.split(key, 16))
        x = particles.x.at[3].set(jnp.nan).at[7].set(jnp.inf)
        particles = env.replace_state_array(particles, x)
        gold = Lorenz96Observation(values=state.x[1::2] + 0.1, sigma_obs=state.sigma_obs)
        w, _ = env.observation_weights(particles, None, gold)
        w = np.asarray(w)
        assert np.all(np.isfinite(w))
        assert w[3] == 0.0 and w[7] == 0.0
        assert np.isclose(w.sum(), 1.0, atol=1e-5)

    def test_uniform_when_all_particles_nonfinite(self, env, key):
        state, _ = env.reset(key)
        particles = jax.vmap(lambda k: env.sample_prior(state, k))(jax.random.split(key, 16))
        particles = env.replace_state_array(particles, jnp.full_like(particles.x, jnp.nan))
        gold = Lorenz96Observation(values=state.x[1::2], sigma_obs=state.sigma_obs)
        w, _ = env.observation_weights(particles, None, gold)
        assert np.allclose(np.asarray(w), np.ones(16) / 16)

    def test_matches_analytic_softmax(self, env, key):
        state, _ = env.reset(key)
        particles = jax.vmap(lambda k: env.sample_prior(state, k))(jax.random.split(key, 32))
        gold = Lorenz96Observation(values=state.x[1::2] + 0.1, sigma_obs=state.sigma_obs)
        w, _ = env.observation_weights(particles, None, gold)
        log_likes = -0.5 * np.sum(
            ((np.asarray(particles.x)[:, 1::2] - np.asarray(gold.values)) / float(gold.sigma_obs))
            ** 2,
            axis=-1,
        )
        expected = np.exp(log_likes - log_likes.max())
        expected /= expected.sum()
        assert np.allclose(w, expected, atol=1e-5)
        assert np.allclose(
            w, jax.nn.softmax(env.observation_log_prob(particles.x, gold)), atol=1e-6
        )


class TestPriorSampler:
    def test_sample_prior_moments(self, env, key):
        state, _ = env.reset(key)
        particles = jax.vmap(lambda k: env.sample_prior(state, k))(jax.random.split(key, 4096))
        mean = np.asarray(jnp.mean(particles.x, axis=0))
        std = np.asarray(jnp.std(particles.x, axis=0))
        assert np.allclose(mean, np.asarray(state.center), atol=0.2)
        assert np.allclose(std, env.sigma_init, atol=0.2)

    def test_pf_reset_uses_prior_sampler(self, env, key):
        state, ts = env.reset(key)
        pf = ParticleFilter(num_particles=1024)
        pset = pf.reset(env, state, ts.observation, key)
        mean = np.asarray(jnp.mean(pset.particles.x, axis=0))
        assert np.allclose(mean, np.asarray(state.center), atol=0.4)
        assert np.allclose(pset.weights, np.ones(1024) / 1024)
        assert np.allclose(np.asarray(pset.particles.forcing), float(state.forcing))


class TestObsPattern:
    def test_fully_observed_env(self, key):
        env = Lorenz96Env(K=6, time_limit=5, obs_sites=tuple(range(6)))
        assert env.num_obs == 6
        state, ts = env.reset(key)
        assert ts.observation.values.shape == (6,)
        assert np.allclose(state.observation_from_state().values, state.x)
        next_state, ts = env.step(state, 0, key)
        assert ts.observation.values.shape == (6,)
        assert np.all(np.abs(np.asarray(ts.observation.values) - np.asarray(next_state.x)) < 7.5)

    def test_custom_sites_thread_through_weights(self, key):
        env = Lorenz96Env(K=6, time_limit=5, obs_sites=(0, 3))
        state, _ = env.reset(key)
        particles = jax.vmap(lambda k: env.sample_prior(state, k))(jax.random.split(key, 16))
        gold = Lorenz96Observation(values=state.x[jnp.array([0, 3])], sigma_obs=state.sigma_obs)
        w, _ = env.observation_weights(particles, None, gold)
        log_likes = -0.5 * np.sum(
            ((np.asarray(particles.x)[:, [0, 3]] - np.asarray(gold.values)) / float(gold.sigma_obs))
            ** 2,
            axis=-1,
        )
        expected = np.exp(log_likes - log_likes.max())
        expected /= expected.sum()
        assert np.allclose(w, expected, atol=1e-5)

    def test_odd_k_requires_explicit_sites(self):
        with pytest.raises(ValueError):
            Lorenz96Env(K=7, time_limit=5)
        env = Lorenz96Env(K=7, time_limit=5, obs_sites=(1, 3, 5))
        assert env.num_obs == 3

    def test_redraw_guards_sigma_dyn_zero(self, key):
        with pytest.raises(ValueError):
            conditional_gaussian_redraw(
                jnp.zeros((4, 8)), jnp.zeros(4), jnp.arange(1, 8, 2), 0.0, 1.0, key
            )
        env = Lorenz96Env(K=8, time_limit=5, sigma_dyn=0.0)
        with pytest.raises(ValueError):
            env.transition_log_prob(jnp.zeros((4, 8)), jnp.zeros((4, 8)))


class TestRejuvenation:
    def test_exact_redraw_moments(self, key):
        K = 8
        sigma_dyn, sigma_obs = 0.3, 0.6
        means = jnp.tile(jnp.arange(K, dtype=jnp.float32)[None], (40000, 1))
        y = jnp.arange(K, dtype=jnp.float32)[1::2] + 0.5
        x = conditional_gaussian_redraw(means, y, jnp.arange(1, K, 2), sigma_dyn, sigma_obs, key)
        x = np.asarray(x)
        post_var = 1.0 / (1.0 / sigma_dyn**2 + 1.0 / sigma_obs**2)
        post_mean = post_var * (np.arange(K)[1::2] / sigma_dyn**2 + np.asarray(y) / sigma_obs**2)
        assert np.allclose(x[:, 0::2].mean(axis=0), np.arange(K)[0::2], atol=0.02)
        assert np.allclose(x[:, 0::2].std(axis=0), sigma_dyn, atol=0.02)
        assert np.allclose(x[:, 1::2].mean(axis=0), post_mean, atol=0.02)
        assert np.allclose(x[:, 1::2].std(axis=0), np.sqrt(post_var), atol=0.02)

    def test_no_duplicates_after_redraw(self, key):
        means = jnp.zeros((512, 8))
        x = np.asarray(
            conditional_gaussian_redraw(means, jnp.zeros(4), jnp.arange(1, 8, 2), 0.25, 1.0, key)
        )
        assert np.unique(x, axis=0).shape[0] == 512

    def test_exact_redraw_matches_rw_mh(self, key):
        K = 8
        sigma_dyn, sigma_obs = 0.3, 0.6
        n = 8192
        means = jnp.tile(jnp.linspace(-1.0, 1.0, K)[None], (n, 1))
        y = jnp.linspace(-1.0, 1.0, K)[1::2] + 0.4
        exact_key, mh_key = jax.random.split(key)
        obs_idx = jnp.arange(1, K, 2)
        exact = np.asarray(
            conditional_gaussian_redraw(means, y, obs_idx, sigma_dyn, sigma_obs, exact_key)
        )
        mh = np.asarray(
            rw_mh_redraw(means, means, y, obs_idx, sigma_dyn, sigma_obs, 400, 0.15, mh_key)
        )
        assert np.allclose(exact.mean(axis=0), mh.mean(axis=0), atol=0.05)
        assert np.allclose(exact.std(axis=0), mh.std(axis=0), atol=0.05)


class TestPfRegNanRobustness:
    def test_nan_particle_does_not_poison_update(self, env, key):
        state, ts = env.reset(key)
        pf = RegularizedParticleFilter(num_particles=64, resample_factor=2.0)
        pset = pf.reset(env, state, ts.observation, key)
        x = pset.particles.x.at[0].set(jnp.nan).at[1].set(jnp.inf)
        pset = ParticleSet(env.replace_state_array(pset.particles, x), pset.weights)
        step_key, update_key = jax.random.split(key)
        _, ts2 = env.step(state, 0, step_key)
        pset2, neff, _, _ = pf.update(
            env, pset, state, ts.observation, 0, ts2.observation, update_key
        )
        assert np.all(np.isfinite(np.asarray(pset2.weights)))
        assert np.all(np.isfinite(np.asarray(pset2.particles.x)))
        assert float(neff) > 0.0

    def test_collapsed_weights_do_not_poison_jitter(self, env, key):
        state, ts = env.reset(key)
        pf = RegularizedParticleFilter(num_particles=256, resample_factor=2.0)
        pset = pf.reset(env, state, ts.observation, key)
        wkey, step_key, update_key = jax.random.split(key, 3)
        w = jax.nn.softmax(1e4 * jax.random.normal(wkey, (256,)))
        pset = ParticleSet(pset.particles, w)
        next_state, ts2 = env.step(state, 0, step_key)
        pset2, _, _, _ = pf.update(env, pset, state, ts.observation, 0, ts2.observation, update_key)
        assert np.all(np.isfinite(np.asarray(pset2.particles.x)))
        assert np.all(np.isfinite(np.asarray(pset2.weights)))


def test_normalize_log_jacobian_matches_change_of_variables():
    # A density in normalized coordinates plus the Jacobian equals the raw-state density:
    # for z = (x - m) / s per coordinate, log p_x(x) = log p_z(z) - K log s.
    env = Lorenz96Env(K=8, time_limit=5, norm_mean=1.5, norm_std=2.5)
    x = jnp.linspace(-3.0, 4.0, 8)[None]
    z = env.normalize_flat(x)
    logp_z = -0.5 * jnp.sum(z**2) - 0.5 * 8 * jnp.log(2 * jnp.pi)  # standard normal in z
    logp_x = logp_z + env.normalize_log_jacobian()
    expected = jax.scipy.stats.norm.logpdf(x[0], loc=1.5, scale=2.5).sum()
    assert jnp.allclose(logp_x, expected, atol=1e-5)
