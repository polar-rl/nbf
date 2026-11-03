import jax
import jax.numpy as jnp
import pytest

from nbf.domains.lorenz96.env import OBS_SENTINEL, Lorenz96Env, Lorenz96Observation
from nbf.filters.proposals import optimal_proposal_particle_set


@pytest.fixture
def env():
    return Lorenz96Env(K=20, time_limit=20, sigma_dyn=0.25, sigma_obs_min=1.0, sigma_obs_max=1.0)


def _particles(env, key, n):
    gen_key, prior_key = jax.random.split(key)
    state0 = env.generator(gen_key)
    return state0, jax.vmap(lambda k: env.sample_prior(state0, k))(jax.random.split(prior_key, n))


def test_op_weights_match_marginal_likelihood(env):
    n = 512
    state0, particles = _particles(env, jax.random.PRNGKey(0), n)
    y = Lorenz96Observation(values=state0.x[env.obs_idx], sigma_obs=jnp.asarray(1.0))
    prior = jnp.full(n, 1.0 / n)
    pset = optimal_proposal_particle_set(env, particles, prior, y, jax.random.PRNGKey(1))

    means = jax.vmap(env.propagate_mean_array)(particles)
    marg_var = env.sigma_dyn**2 + 1.0
    log_w = -0.5 * jnp.sum((means[:, env.obs_idx] - y.values) ** 2, axis=-1) / marg_var
    expected = jax.nn.softmax(log_w)
    assert jnp.allclose(pset.weights, expected, atol=1e-5)


def test_op_redraw_conditions_observed_sites(env):
    n = 8192
    state0, particles = _particles(env, jax.random.PRNGKey(2), n)
    y = Lorenz96Observation(values=state0.x[env.obs_idx], sigma_obs=jnp.asarray(1.0))
    prior = jnp.full(n, 1.0 / n)
    pset = optimal_proposal_particle_set(env, particles, prior, y, jax.random.PRNGKey(3))

    means = jax.vmap(env.propagate_mean_array)(particles)
    moved = env.state_array(pset.particles)
    sd2, so2 = env.sigma_dyn**2, 1.0
    post_var = 1.0 / (1.0 / sd2 + 1.0 / so2)
    post_mean = post_var * (means[:, env.obs_idx] / sd2 + y.values / so2)
    resid = moved[:, env.obs_idx] - post_mean
    assert jnp.abs(jnp.mean(resid)) < 0.02
    assert jnp.abs(jnp.std(resid) - jnp.sqrt(post_var)) < 0.02

    unobs = jnp.setdiff1d(jnp.arange(env.K), env.obs_idx)
    resid_u = moved[:, unobs] - means[:, unobs]
    assert jnp.abs(jnp.std(resid_u) - env.sigma_dyn) < 0.02


def test_op_ess_beats_blind_weighting(env):
    n = 2048
    state0, particles = _particles(env, jax.random.PRNGKey(4), n)
    sharp = Lorenz96Observation(values=state0.x[env.obs_idx], sigma_obs=jnp.asarray(0.5))
    prior = jnp.full(n, 1.0 / n)
    pset = optimal_proposal_particle_set(env, particles, prior, sharp, jax.random.PRNGKey(5))

    means = jax.vmap(env.propagate_mean_array)(particles)
    raw = means + env.sigma_dyn * jax.random.normal(jax.random.PRNGKey(6), means.shape)
    raw_log_w = -0.5 * jnp.sum(((raw[:, env.obs_idx] - sharp.values) / 0.5) ** 2, axis=-1)
    raw_w = jax.nn.softmax(raw_log_w)

    ess = lambda w: 1.0 / jnp.sum(w**2)
    assert ess(pset.weights) > ess(raw_w)


def test_sentinel_observation_falls_back_to_blind(env):
    n = 256
    _, particles = _particles(env, jax.random.PRNGKey(7), n)
    sentinel = Lorenz96Observation(
        values=jnp.full(env.num_obs, OBS_SENTINEL), sigma_obs=jnp.asarray(1.0)
    )
    prior = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(8), (n,)))
    pset = optimal_proposal_particle_set(env, particles, prior, sentinel, jax.random.PRNGKey(9))
    assert jnp.allclose(pset.weights, prior, atol=1e-6)
