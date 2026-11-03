import jax
import jax.numpy as jnp
import pytest

from nbf.domains.pursuit_evasion.env import (
    SPEED,
    L,
    PursuitEvasionEnv,
    PursuitEvasionObservation,
    sensor_ranges,
)
from nbf.domains.pursuit_evasion.eval.base import PEObservationModel, make_env_kn, make_pe_gmf
from nbf.domains.pursuit_evasion.eval.samples import gmf_samples_density_stream_batch
from nbf.domains.pursuit_evasion.policies import make_goal_policy, make_random_walk_policy
from nbf.eval.base import rollout_episodes


def _env(k=3, m=2, policy=None):
    return PursuitEvasionEnv(
        num_evaders=k, time_limit=6, policy=policy, shuffle_observations=True, num_sensors=m
    )


def test_observation_model_orders_by_component_mean():
    k, m = 3, 2
    env = _env(k, m)
    om = PEObservationModel(env)
    mu = jax.random.uniform(jax.random.PRNGKey(0), (2 * k,), minval=1.0, maxval=9.0)
    sigmas = mu[None] + 0.01 * jax.random.normal(jax.random.PRNGKey(1), (2 * k + 1, 2 * k))
    pred = om.predicted(sigmas, mu)
    assert pred.shape == (2 * k + 1, m * k)
    blocks = pred.reshape(-1, m, k)
    assert bool(jnp.all(blocks[:, :, 1:] >= blocks[:, :, :-1]))
    far = mu[None] + jnp.array([[3.0, -3.0, 0.0, 0.0, 0.0, 0.0]])
    r_far = sensor_ranges(far.reshape(1, k, 2), env.sensors)[0]  # (m, k)
    order = jnp.argsort(sensor_ranges(mu.reshape(k, 2), env.sensors), axis=-1)
    expect = jnp.take_along_axis(r_far, order, axis=-1).reshape(m * k)
    assert jnp.allclose(om.predicted(far, mu)[0], expect)
    obs = PursuitEvasionObservation(
        ranges=jnp.array([3.0, 1.0, 2.0, 6.0, 4.0, 5.0]), observation_sigma=jnp.asarray(0.2)
    )
    assert jnp.allclose(om.vector(obs), jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert float(om.noise_std(obs)) == pytest.approx(0.2)


def test_update_uses_episode_noise():
    k, m = 2, 1
    env = _env(k, m)
    gmf = make_pe_gmf(M=1, env=env, heading_draws=0)
    mu = jnp.array([3.0, 3.0, 7.0, 6.0])
    sigma = 0.5 * jnp.eye(2 * k)
    ranges = env.true_ranges(mu.reshape(k, 2))
    traces = []
    for s in (0.05, 0.5):
        obs = PursuitEvasionObservation(ranges=ranges, observation_sigma=jnp.asarray(s))
        _, sg_post = gmf.ukf.update(mu, sigma, obs)
        traces.append(float(jnp.trace(sg_post)))
    assert traces[0] < traces[1] < float(jnp.trace(sigma))


def test_heading_averaged_forecast_for_random_walk():
    k = 1
    env = _env(k, 1, policy=make_random_walk_policy())
    gmf = make_pe_gmf(M=1, env=env, heading_draws=64)
    mu = jnp.array([5.0, 8.5])
    sigma = 0.01 * jnp.eye(2)
    mu_pred, sg_pred = gmf.ukf.predict(mu, sigma, env.policy, env.obstacles, jax.random.PRNGKey(3))
    assert float(jnp.linalg.norm(mu_pred - mu)) < 0.3 * SPEED
    assert float(jnp.trace(sg_pred)) > float(jnp.trace(gmf.ukf.Q)) + 0.5 * SPEED**2
    env_g = _env(k, 1, policy=make_goal_policy(jnp.array([9.0, 9.0])))
    gmf_g = make_pe_gmf(M=1, env=env_g, heading_draws=64)
    mu_g, _ = gmf_g.ukf.predict(
        jnp.array([5.0, 8.5]), sigma, env_g.policy, env_g.obstacles, jax.random.PRNGKey(4)
    )
    assert 0.5 * SPEED < float(jnp.linalg.norm(mu_g - jnp.array([5.0, 8.5]))) <= SPEED + 1e-4


def test_step_skips_sentinel_observation():
    k, m = 2, 2
    env = _env(k, m, policy=make_random_walk_policy())
    gmf = make_pe_gmf(M=4, env=env)
    w0, mu0, sg0 = gmf.init(key=jax.random.PRNGKey(5))
    sentinel = PursuitEvasionObservation(
        ranges=jnp.full(m * k, -1.0), observation_sigma=jnp.asarray(0.3)
    )
    key = jax.random.PRNGKey(6)
    w, mu, sg = gmf.step(w0, mu0, sg0, env.policy, env.obstacles, sentinel, key)
    w_p, mu_p, sg_p = gmf.predict(w0, mu0, sg0, env.policy, env.obstacles, key)
    assert jnp.allclose(w, w_p) and jnp.allclose(mu, mu_p) and jnp.allclose(sg, sg_p)


def test_density_stream_runs_on_multisensor_shuffled_env():
    k, m, T, B = 2, 2, 5, 2
    env = make_env_kn(k, T, (make_random_walk_policy(),) * k, num_sensors=m)
    state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b = rollout_episodes(
        env, T, B, jax.random.PRNGKey(7)
    )
    gmf = make_pe_gmf(M=4, env=env)
    samps, lp = gmf_samples_density_stream_batch(
        env,
        gmf,
        states_next_b,
        obs_next_b,
        jax.random.split(jax.random.PRNGKey(8), B),
        jax.random.split(jax.random.PRNGKey(9), B * T).reshape(B, T, -1),
        16,
    )
    assert samps.shape == (B, T, 16, k, 2) and lp.shape == (B, T)
    assert bool(jnp.all(jnp.isfinite(lp))) and bool(jnp.all(jnp.isfinite(samps)))
    assert bool(jnp.all(jnp.abs(samps) < 3 * L))
