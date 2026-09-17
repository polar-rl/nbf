"""The proposal interface: the bootstrap step is generate + ParticleSet.simulate + likelihood
features, the optimal proposal is the same filter with a different proposal, and the filter's
public call surface (positional constructor, update/step/sample) holds."""

import os

import jax
import jax.numpy as jnp
import pytest
from embedex.distributions.builder import create_model

from nbf.domains.lorenz96.env import build_lorenz96_env
from nbf.domains.lorenz96.train import load_config, simulate_episodes
from nbf.filters.likelihood import likelihood_feature_dim, likelihood_features
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.particle import ParticleSet
from nbf.filters.proposals import (
    BootstrapProposal,
    OptimalProposal,
    make_proposal,
    optimal_proposal_particle_set,
)

SMOKE = os.path.join(
    os.path.dirname(__file__), '..', '..', 'nbf', 'domains', 'lorenz96', 'configs',
    'smoke_l96.yaml',
)


@pytest.fixture(scope='module')
def setup():
    cfg = load_config(SMOKE, ['training.rollout_num_particles=32'])
    cfg.data.norm_mean, cfg.data.norm_std = 2.5, 3.9
    env = build_lorenz96_env(cfg.data)
    cfg.model.output_dim = cfg.data.K
    cfg.model.input_dim = cfg.data.K + likelihood_feature_dim(env)
    model = create_model(cfg.model, dequantize=False, key=jax.random.PRNGKey(0))
    eps = simulate_episodes(env, 1, jax.random.PRNGKey(1))
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
    # so exact equality is only expected between two jitted paths (see
    # scripts/check_filter_bitwise.py for the cross-commit bitwise guard).
    return all(
        bool(jnp.allclose(x, y, atol=atol, rtol=1e-5))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
    )


def test_bootstrap_step_is_generate_simulate_featurize(setup):
    # The step, inlined: same key split, uniform weights into ParticleSet.simulate, then the
    # likelihood features of the proposed particles against the next observation.
    env, model, s0, o0, st, op, on = setup
    filt = NeuralBayesianFilter(model, 32)
    emb = filt.reset(env, s0, o0, jax.random.PRNGKey(2))
    key = jax.random.PRNGKey(3)

    sample_key, simulate_key = jax.random.split(key, 2)
    particles = filt.sample(emb, st, 31, sample_key)
    particles = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y[jnp.newaxis]], axis=0), particles, st
    )
    expected, simulation_fallback = ParticleSet(particles, jnp.full(32, 1 / 32)).simulate(
        env, 0, on, simulate_key
    )
    expected_features = likelihood_features(env, expected.particles, on)

    got, got_sampling, got_simulation = filt.step(env, emb, st, op, 0, on, key)
    assert _tree_close(got.particles, expected.particles, atol=1e-6)
    assert _tree_close(got.weights, expected.weights, atol=1e-6)
    assert _tree_close(got.features, expected_features, atol=1e-6)
    assert got.features.shape == (32, env.K)
    assert not bool(got_sampling) and bool(got_simulation == simulation_fallback)


def test_repr_names_the_proposal(setup):
    _, model, *_ = setup
    assert repr(NeuralBayesianFilter(model, 32)) == 'NBF (32)'
    assert repr(NeuralBayesianFilter(model, 32, proposal=OptimalProposal())) == 'NBF-OP (32)'


def test_optimal_step_uses_optimal_proposal_particle_set(setup):
    env, model, s0, o0, st, op, on = setup
    filt = NeuralBayesianFilter(model, 32, proposal=OptimalProposal())
    emb = filt.reset(env, s0, o0, jax.random.PRNGKey(6))
    key = jax.random.PRNGKey(7)
    sample_key, propose_key = jax.random.split(key, 2)
    particles = filt.sample(emb, st, 31, sample_key)
    particles = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y[jnp.newaxis]], axis=0), particles, st
    )
    expected = optimal_proposal_particle_set(
        env, particles, jnp.full(32, 1 / 32), on, propose_key
    )
    got, _, fallback = filt.step(env, emb, st, op, 0, on, key)
    assert _tree_close(got.particles, expected.particles)
    assert _tree_close(got.weights, expected.weights)
    assert not bool(fallback)


def test_public_call_surface(setup):
    # Downstream users call: positional constructor, sample, reset, update, step; update
    # returns a 4-tuple headed by the embedding; the step's set carries its features and
    # the encoder input is state features next to likelihood features under uniform weights.
    env, model, s0, o0, st, op, on = setup
    filt = NeuralBayesianFilter(model, 32)
    assert isinstance(filt.proposal, BootstrapProposal)
    emb = filt.reset(env, s0, o0, jax.random.PRNGKey(8))
    out = filt.update(env, emb, st, op, 0, on, jax.random.PRNGKey(9))
    assert len(out) == 4 and out[0].shape == emb.shape
    pset, *_ = filt.step(env, emb, st, op, 0, on, jax.random.PRNGKey(10))
    assert pset.weights.shape == (32,) and pset.features.shape == (32, env.K)
    feats, ws = filt.encoder_input(pset)
    assert feats.shape == (32, 2 * env.K)
    assert jnp.allclose(ws, 1 / 32)
    assert filt.sample(emb, st, 5, jax.random.PRNGKey(12)).x.shape == (5, env.K)


def test_make_proposal():
    assert isinstance(make_proposal('bootstrap'), BootstrapProposal)
    assert isinstance(make_proposal('optimal'), OptimalProposal)
    with pytest.raises(ValueError):
        make_proposal('nope')


def test_optimal_proposal_needs_env_hooks(setup):
    env, model, s0, o0, st, op, on = setup

    class NoHooks:
        pass

    with pytest.raises(ValueError):
        OptimalProposal().propose(NoHooks(), None, None, 0, on, jax.random.PRNGKey(0))
