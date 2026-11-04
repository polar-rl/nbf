"""The NBF step is generate + ParticleSet.simulate + likelihood features, and the filter's
public call surface (positional constructor, update/step/sample) holds."""

import jax
import jax.numpy as jnp
import pytest
from embedex.distributions.builder import create_model

from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.particle import ParticleSet


@pytest.fixture(scope='module')
def setup(domains):
    lorenz96 = domains['lorenz96']
    cfg = lorenz96.config(['training.rollout_num_particles=32'])
    cfg.data.norm_mean, cfg.data.norm_std = 2.5, 3.9
    env = lorenz96.build_env(cfg.data)
    cfg.model.input_dim = cfg.data.K
    cfg.model.likelihood_dim = env.likelihood_feature_dim
    model = create_model(cfg.model, dequantize=False, key=jax.random.PRNGKey(0))
    eps = lorenz96.module.simulate_episodes(env, 1, jax.random.PRNGKey(1))
    one = lambda t: jax.tree.map(lambda x: x[0], t)  # noqa: E731
    st, op, on = (
        jax.tree.map(lambda x: x[0, 0], t) for t in (eps.states_next, eps.obs_prev, eps.obs_next)
    )
    return env, model, one(eps.state0), one(eps.obs0), st, op, on


def _tree_equal(a, b) -> bool:
    return all(
        bool(jnp.array_equal(x, y))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    )


def _tree_close(a, b, atol=1e-5) -> bool:
    # step() runs under filter_jit and an eager reference does not fuse the same way,
    # so exact equality is only expected between two jitted paths.
    return all(
        bool(jnp.allclose(x, y, atol=atol, rtol=1e-5))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    )


def test_bootstrap_step_is_simulate_then_featurize(setup):
    # The step, inlined: a uniformly weighted set into ParticleSet.simulate, then the
    # likelihood features of the proposed particles against the next observation.
    env, model, s0, o0, st, op, on = setup
    filt = NeuralBayesianFilter(model, 32)
    emb = filt.reset(env, s0, o0, jax.random.PRNGKey(2))
    particles = filt.sample(emb, st, 32, jax.random.PRNGKey(3))
    key = jax.random.PRNGKey(4)

    expected, simulation_fallback = ParticleSet(particles, jnp.full(32, 1 / 32)).simulate(
        env, 0, on, key
    )
    expected_features = env.likelihood_features(expected.particles, on, expected.weights)

    got, got_simulation = filt.step(env, particles, 0, on, key)
    assert _tree_close(got.particles, expected.particles, atol=1e-6)
    assert _tree_close(got.weights, expected.weights, atol=1e-6)
    assert _tree_close(got.features, expected_features, atol=1e-6)
    assert got.features.shape == (32, env.K)
    assert bool(got_simulation == simulation_fallback)


def test_update_particles_generates_the_whole_set(setup):
    env, model, s0, o0, st, op, on = setup
    filt = NeuralBayesianFilter(model, 32)
    emb = filt.reset(env, s0, o0, jax.random.PRNGKey(2))
    key = jax.random.PRNGKey(3)
    sample_key, step_key = jax.random.split(key, 2)
    particles = filt.sample(emb, st, 32, sample_key)
    expected, _ = filt.step(env, particles, 0, on, step_key)

    got, neff, sampling_fallback, _ = filt.update_particles(env, emb, st, op, 0, on, key)
    assert _tree_close(got.particles, expected.particles, atol=1e-6)
    assert _tree_close(got.features, expected.features, atol=1e-6)
    assert bool(jnp.isclose(neff, expected.neff()))
    assert not bool(sampling_fallback)


def test_repr(setup):
    _, model, *_ = setup
    assert repr(NeuralBayesianFilter(model, 32)) == 'NBF (32)'


def test_public_call_surface(setup):
    # Downstream users call: positional constructor, sample, reset, update, step; update
    # returns a 4-tuple headed by the embedding; step takes the particle set (any size) and
    # returns it with its features; the encoder input is the states next to their
    # likelihood features.
    env, model, s0, o0, st, op, on = setup
    filt = NeuralBayesianFilter(model, 32)
    emb = filt.reset(env, s0, o0, jax.random.PRNGKey(8))
    out = filt.update(env, emb, st, op, 0, on, jax.random.PRNGKey(9))
    assert len(out) == 4 and out[0].shape == emb.shape
    particles = filt.sample(emb, st, 5, jax.random.PRNGKey(12))
    assert particles.x.shape == (5, env.K)
    pset, _ = filt.step(env, particles, 0, on, jax.random.PRNGKey(10))
    assert pset.weights.shape == (5,) and pset.features.shape == (5, env.K)
    states, lik = filt.encoder_input(pset)
    assert states.shape == (5, env.K) and lik.shape == (5, env.K)
