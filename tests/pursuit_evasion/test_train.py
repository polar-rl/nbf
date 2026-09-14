"""Pursuit-evasion on-policy trainer: per-episode policies, rollout layout, one train step."""

import os

import jax
import jax.numpy as jnp
import pytest
from embedex.distributions.builder import create_model
from embedex.distributions.training import init_optimizer

from nbf.domains.pursuit_evasion.train import (
    build_env,
    env_with_policies,
    load_config,
    sample_policy_batch,
    simulate_episodes,
)
from nbf.train import collect_rollout, gold_loss, train_step

SMOKE = os.path.join(
    os.path.dirname(__file__),
    '..',
    '..',
    'nbf',
    'domains',
    'pursuit_evasion',
    'configs',
    'smoke_k1.yaml',
)


def _setup(k: int):
    cfg = load_config(SMOKE, [f'data.num_evaders={k}', 'training.episodes_per_rollout=4'])
    env = build_env(cfg.data)
    cfg.model.input_dim = 2 * k
    cfg.model.likelihood_dim = env.likelihood_feature_dim
    model = create_model(cfg.model, dequantize=False, key=jax.random.PRNGKey(0))
    return cfg, env, model


@pytest.fixture(scope='module')
def setup_k1():
    return _setup(1)


@pytest.fixture(scope='module')
def setup_k2():
    return _setup(2)


def test_policy_batch_is_stacked_per_evader():
    pols = sample_policy_batch(5, 2, jax.random.PRNGKey(1))
    assert len(pols) == 2
    for p in pols:
        assert p.kind.shape == (5,)
        assert p.point.shape == (5, 2)


def test_env_axes_mark_only_policy_leaves(setup_k2):
    _, env, _ = setup_k2
    pols = sample_policy_batch(4, 2, jax.random.PRNGKey(2))
    env_b, env_axes = env_with_policies(env, pols)
    assert len(env_b.policy) == 2 and env_b.policy[0].kind.shape == (4,)
    assert all(a == 0 for a in jax.tree.leaves(env_axes.policy))
    assert env_axes.obstacles is None and env_axes.generator.obstacles is None


@pytest.mark.parametrize('fixture', ['setup_k1', 'setup_k2'])
def test_simulate_episodes_layout(fixture, request):
    cfg, env, _ = request.getfixturevalue(fixture)
    B, T, k = 3, env.time_limit, env.num_evaders
    eps = simulate_episodes(env, B, jax.random.PRNGKey(3))
    assert eps.x_norm.shape == (B, T + 1, 2 * k)
    assert eps.states_next.positions.shape == (B, T, k, 2)
    assert eps.obs_next.ranges.shape == (B, T, k)
    assert bool(jnp.all(jnp.abs(eps.x_norm) <= 1.0 + 1e-5))
    # Different episodes were simulated under different policies.
    kinds = eps.env.policy[0].kind
    assert kinds.shape == (B,)
    # obs0 carries the no-observation sentinel, later observations are real ranges.
    assert bool(jnp.all(eps.obs0.ranges < 0.0)) and bool(jnp.all(eps.obs_next.ranges >= 0.0))


@pytest.mark.parametrize('fixture', ['setup_k1', 'setup_k2'])
def test_collect_rollout_shapes(fixture, request):
    cfg, env, model = request.getfixturevalue(fixture)
    tcfg = cfg.training
    B, T, n, D = (
        tcfg.episodes_per_rollout,
        env.time_limit,
        tcfg.rollout_num_particles,
        2 * env.num_evaders,
    )
    buffer, stats = collect_rollout(env, model, tcfg, simulate_episodes, jax.random.PRNGKey(4))
    assert buffer['states'].shape == (B * (T + 1), n, D)
    assert buffer['likelihood'].shape == (B * (T + 1), n, env.num_evaders)
    assert buffer['x_true'].shape == (B * (T + 1), D)
    assert bool(jnp.all(jnp.isfinite(buffer['states'])))
    assert bool(jnp.all(jnp.isfinite(buffer['likelihood'])))
    assert 0.0 <= stats['neff_over_n'] <= 1.0


def test_train_step_reduces_loss_on_fixed_batch(setup_k1):
    cfg, env, model = setup_k1
    tcfg = cfg.training
    buffer, _ = collect_rollout(env, model, tcfg, simulate_episodes, jax.random.PRNGKey(5))
    idx = jnp.arange(min(16, buffer['x_true'].shape[0]))
    f, w, x = buffer['states'][idx], buffer['likelihood'][idx], buffer['x_true'][idx]
    key = jax.random.PRNGKey(6)
    tcfg.warmup_steps, tcfg.learning_rate, tcfg.num_iterations = 0, 1e-3, 100
    optimizer, opt_state = init_optimizer(model, tcfg)
    before = float(gold_loss(model, f, w, x, key)[0])
    assert jnp.isfinite(before)
    for _ in range(30):
        model, opt_state, loss, _ = train_step(model, optimizer, opt_state, f, w, x, key)
    assert float(gold_loss(model, f, w, x, key)[0]) < before
