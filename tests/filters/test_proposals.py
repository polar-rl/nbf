"""The proposal interface: bootstrap reproduces the pre-interface filter bit for bit,
the optimal proposal is the same filter with a different proposal, and BETS's call
surface (positional constructor, update/step/maybe_resample/sample) is unchanged."""

import os

import jax
import jax.numpy as jnp
import pytest
from embedex.distributions.builder import create_model

from nbf.domains.lorenz96.env import build_lorenz96_env
from nbf.domains.lorenz96.train import load_config, simulate_episodes
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.nbf_op import OptimalProposalNeuralBayesianFilter
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
    cfg.model.input_dim = cfg.model.output_dim = cfg.data.K
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


def test_bootstrap_step_matches_pre_interface_algorithm(setup):
    # The old NeuralBayesianFilter.step, inlined: same key split, same ParticleSet.simulate.
    env, model, s0, o0, st, op, on = setup
    filt = NeuralBayesianFilter(model, 32)
    emb = filt.reset(env, s0, o0, jax.random.PRNGKey(2))
    key = jax.random.PRNGKey(3)

    sample_key, simulate_key = jax.random.split(key, 2)
    particles = filt.sample(emb, st, 31, sample_key)
    particles = jax.tree.map(
        lambda x, y: jnp.concatenate([x, y[jnp.newaxis]], axis=0), particles, st
    )
    weights, sampling_fallback = filt.compare_state_obs(env, particles, op)
    expected, simulation_fallback = ParticleSet(particles, weights).simulate(
        env, 0, on, simulate_key
    )

    got, got_sampling, got_simulation = filt.step(env, emb, st, op, 0, on, key)
    assert _tree_close(got, expected, atol=1e-6)
    assert bool(got_sampling == sampling_fallback) and bool(got_simulation == simulation_fallback)


def test_optimal_class_is_filter_with_optimal_proposal(setup):
    env, model, s0, o0, st, op, on = setup
    a = OptimalProposalNeuralBayesianFilter(model, 32)
    b = NeuralBayesianFilter(model, 32, proposal=OptimalProposal())
    emb = a.reset(env, s0, o0, jax.random.PRNGKey(4))
    key = jax.random.PRNGKey(5)
    assert _tree_equal(a.update(env, emb, st, op, 0, on, key), b.update(env, emb, st, op, 0, on, key))
    assert repr(a) == 'NBF-OP (32)' and repr(NeuralBayesianFilter(model, 32)) == 'NBF (32)'


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
    weights, _ = filt.compare_state_obs(env, particles, op)
    expected = optimal_proposal_particle_set(env, particles, weights, on, propose_key)
    got, _, fallback = filt.step(env, emb, st, op, 0, on, key)
    assert _tree_close(got, expected)
    assert not bool(fallback)


def test_bets_call_surface(setup):
    # What bets imports and calls: positional constructor, sample, reset, update, step,
    # maybe_resample; update returns a 4-tuple headed by the embedding.
    env, model, s0, o0, st, op, on = setup
    filt = NeuralBayesianFilter(model, 32)
    assert isinstance(filt.proposal, BootstrapProposal)
    emb = filt.reset(env, s0, o0, jax.random.PRNGKey(8))
    out = filt.update(env, emb, st, op, 0, on, jax.random.PRNGKey(9))
    assert len(out) == 4 and out[0].shape == emb.shape
    pset, *_ = filt.step(env, emb, st, op, 0, on, jax.random.PRNGKey(10))
    pset = filt.maybe_resample(pset, jax.random.PRNGKey(11))
    assert pset.weights.shape == (32,)
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
