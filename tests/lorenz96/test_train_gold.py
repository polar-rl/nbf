import os

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from embedex.distributions.builder import create_model
from embedex.distributions.training import init_optimizer

from nbf.domains.lorenz96.env import build_lorenz96_env
from nbf.domains.lorenz96.train_gold import (
    collect_rollout,
    gold_loss,
    load_config,
    make_filter,
    rollout_model_from,
    rollout_particle_sets,
    simulate_episodes,
    train_step,
)
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.nbf_op import OptimalProposalNeuralBayesianFilter

SMOKE = os.path.join(
    os.path.dirname(__file__),
    '..',
    '..',
    'nbf',
    'domains',
    'lorenz96',
    'configs',
    'smoke_l96_gold.yaml',
)


@pytest.fixture(scope='module')
def setup():
    cfg = load_config(
        SMOKE, ['training.rollout_num_particles=32', 'training.episodes_per_rollout=3']
    )
    cfg.data.norm_mean, cfg.data.norm_std = 2.5, 3.9
    env = build_lorenz96_env(cfg.data)
    cfg.model.input_dim = cfg.model.output_dim = cfg.data.K
    model = create_model(cfg.model, dequantize=False, key=jax.random.PRNGKey(0))
    return cfg, env, model


def test_config_rejects_unknown_rollout_filter():
    with pytest.raises(ValueError):
        load_config(SMOKE, ['training.rollout_filter=bogus'])


@pytest.mark.parametrize('cls', [NeuralBayesianFilter, OptimalProposalNeuralBayesianFilter])
def test_update_is_update_particles_then_embed(setup, cls):
    _, env, model = setup
    filt = cls(model, num_particles=32)
    state0, obs0, states_next, obs_prev, obs_next, _ = simulate_episodes(
        env, 1, jax.random.PRNGKey(1)
    )
    one = lambda t: jax.tree.map(lambda x: x[0], t)  # noqa: E731
    s0, o0 = one(state0), one(obs0)
    st, op, on = (jax.tree.map(lambda x: x[0, 0], t) for t in (states_next, obs_prev, obs_next))

    rkey, ukey = jax.random.split(jax.random.PRNGKey(2))
    emb0 = filt.reset(env, s0, o0, rkey)
    feats0, w0 = filt.reset_particles(env, s0, o0, rkey)
    assert jnp.allclose(emb0, model.embed(feats0, w0)[0])
    assert jnp.allclose(jnp.sum(w0), 1.0, atol=1e-5)

    emb1, neff, *_ = filt.update(env, emb0, st, op, 0, on, ukey)
    pset, neff2, *_ = filt.update_particles(env, emb0, st, op, 0, on, ukey)
    assert jnp.allclose(emb1, filt.embed_particles(pset))
    assert jnp.allclose(neff, neff2)


def test_rollout_shapes_and_normalization(setup):
    cfg, env, model = setup
    tcfg = cfg.training
    B, T, n, K = tcfg.episodes_per_rollout, env.time_limit, tcfg.rollout_num_particles, env.K
    for kind in ('nbf', 'nbf_op'):
        buffer, stats = collect_rollout(env, model, kind, tcfg, jax.random.PRNGKey(3))
        assert buffer['features'].shape == (B * (T + 1), n, K)
        assert buffer['weights'].shape == (B * (T + 1), n)
        assert buffer['x_true'].shape == (B * (T + 1), K)
        assert bool(jnp.all(jnp.isfinite(buffer['features'])))
        assert jnp.allclose(jnp.sum(buffer['weights'], axis=-1), 1.0, atol=1e-4)
        assert 0.0 <= stats['neff_over_n'] <= 1.0
        assert 0.0 <= stats['resample_frac'] <= 1.0


def test_rollout_row_zero_is_the_prior_set(setup):
    cfg, env, model = setup
    n = cfg.training.rollout_num_particles
    filt = make_filter(model, 'nbf', n)
    state0, obs0, states_next, obs_prev, obs_next, _ = simulate_episodes(
        env, 2, jax.random.PRNGKey(4)
    )
    rkeys = jax.random.split(jax.random.PRNGKey(5), 2)
    skeys = jax.random.split(jax.random.PRNGKey(6), 2 * env.time_limit).reshape(
        2, env.time_limit, -1
    )
    feats, ws, neffs = rollout_particle_sets(
        env, filt, state0, obs0, states_next, obs_prev, obs_next, rkeys, skeys
    )
    assert neffs.shape == (2, env.time_limit)
    f0, w0 = filt.reset_particles(
        env, jax.tree.map(lambda x: x[1], state0), jax.tree.map(lambda x: x[1], obs0), rkeys[1]
    )
    assert jnp.allclose(feats[1, 0], f0)
    assert jnp.allclose(ws[1, 0], w0)


def test_rollout_model_shares_params_and_changes_steps(setup):
    cfg, env, model = setup
    rmodel = rollout_model_from(model, cfg.model, 3, False, jax.random.PRNGKey(7))
    assert rmodel.generator.num_steps == 3
    assert model.generator.num_steps == cfg.model.generator.fm_num_steps != 3
    a = jax.tree.leaves(eqx.filter(model.generator.velocity_net, eqx.is_array))
    b = jax.tree.leaves(eqx.filter(rmodel.generator.velocity_net, eqx.is_array))
    assert all(jnp.array_equal(x, y) for x, y in zip(a, b, strict=True))
    xs = jax.random.normal(jax.random.PRNGKey(8), (16, env.K))
    w = jnp.full(16, 1 / 16)
    assert jnp.allclose(model.embed(xs, w)[0], rmodel.embed(xs, w)[0])
    # unchanged steps -> same object, no rebuild
    same = rollout_model_from(model, cfg.model, cfg.model.generator.fm_num_steps, False, None)
    assert same is model


def test_train_step_reduces_gold_loss_on_fixed_batch(setup):
    cfg, env, model = setup
    tcfg = cfg.training
    buffer, _ = collect_rollout(env, model, 'nbf_op', tcfg, jax.random.PRNGKey(9))
    idx = jnp.arange(min(16, buffer['x_true'].shape[0]))
    f, w, x = buffer['features'][idx], buffer['weights'][idx], buffer['x_true'][idx]
    key = jax.random.PRNGKey(10)
    tcfg.warmup_steps, tcfg.learning_rate, tcfg.num_iterations = 0, 1e-3, 100
    optimizer, opt_state = init_optimizer(model, tcfg)
    before = float(gold_loss(model, f, w, x, key)[0])
    assert jnp.isfinite(before)
    for _ in range(30):
        model, opt_state, loss, _ = train_step(model, optimizer, opt_state, f, w, x, key)
    after = float(gold_loss(model, f, w, x, key)[0])
    assert after < before
