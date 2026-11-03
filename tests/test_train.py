import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from embedex.distributions.builder import create_model
from embedex.distributions.training import init_optimizer

from nbf.domains.lorenz96.train import simulate_episodes
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.train import (
    collect_rollout,
    gold_loss,
    make_filter,
    rollout_model_from,
    rollout_particle_sets,
    train_step,
    validate,
)


@pytest.fixture(scope='module')
def setup(domains):
    lorenz96 = domains['lorenz96']
    cfg = lorenz96.config(['training.rollout_num_particles=32', 'training.episodes_per_rollout=4'])
    cfg.data.norm_mean, cfg.data.norm_std = 2.5, 3.9
    env = lorenz96.build_env(cfg.data)
    cfg.model.input_dim = cfg.data.K
    cfg.model.likelihood_dim = env.likelihood_feature_dim
    model = create_model(cfg.model, dequantize=False, key=jax.random.PRNGKey(0))
    return cfg, env, model


def test_update_is_update_particles_then_embed(setup):
    _, env, model = setup
    filt = NeuralBayesianFilter(model, 32)
    eps = simulate_episodes(env, 1, jax.random.PRNGKey(1))
    one = lambda t: jax.tree.map(lambda x: x[0], t)  # noqa: E731
    s0, o0 = one(eps.state0), one(eps.obs0)
    st, op, on = (
        jax.tree.map(lambda x: x[0, 0], t) for t in (eps.states_next, eps.obs_prev, eps.obs_next)
    )

    rkey, ukey = jax.random.split(jax.random.PRNGKey(2))
    emb0 = filt.reset(env, s0, o0, rkey)
    feats0, w0 = filt.reset_particles(env, s0, o0, rkey)
    assert jnp.allclose(emb0, model.embed(feats0, w0)[0])
    assert w0.shape == (feats0.shape[0], env.K)

    emb1, neff, *_ = filt.update(env, emb0, st, op, 0, on, ukey)
    pset, neff2, *_ = filt.update_particles(env, emb0, st, op, 0, on, ukey)
    assert jnp.allclose(emb1, filt.embed_particles(pset))
    assert jnp.allclose(neff, neff2)


def test_rollout_shapes_and_normalization(setup):
    cfg, env, model = setup
    tcfg = cfg.training
    B, T, n, K = tcfg.episodes_per_rollout, env.time_limit, tcfg.rollout_num_particles, env.K
    buffer, stats = collect_rollout(env, model, tcfg, simulate_episodes, jax.random.PRNGKey(3))
    assert buffer['states'].shape == (B * (T + 1), n, K)
    assert buffer['likelihood'].shape == (B * (T + 1), n, K)
    assert buffer['x_true'].shape == (B * (T + 1), K)
    assert bool(jnp.all(jnp.isfinite(buffer['states'])))
    assert bool(jnp.all(jnp.isfinite(buffer['likelihood'])))
    assert 0.0 <= stats['neff_over_n'] <= 1.0
    assert 0.0 <= stats['resample_frac'] <= 1.0


def test_rollout_row_zero_is_the_prior_set(setup):
    cfg, env, model = setup
    n = cfg.training.rollout_num_particles
    filt = make_filter(model, n)
    eps = simulate_episodes(env, 2, jax.random.PRNGKey(4))
    state0, obs0, states_next, obs_prev, obs_next = (
        eps.state0,
        eps.obs0,
        eps.states_next,
        eps.obs_prev,
        eps.obs_next,
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
    lik = jnp.zeros((16, env.K))
    assert jnp.allclose(model.embed(xs, lik)[0], rmodel.embed(xs, lik)[0])
    # unchanged steps -> same object, no rebuild
    same = rollout_model_from(model, cfg.model, cfg.model.generator.fm_num_steps, False, None)
    assert same is model


def test_train_step_micro_batches(setup):
    """Accumulated gradients train the same way as the one-shot step and are
    deterministic; the CFM draws differ between the two paths, so compare progress, not
    the exact numbers."""
    cfg, env, model = setup
    tcfg = cfg.training
    buffer, _ = collect_rollout(env, model, tcfg, simulate_episodes, jax.random.PRNGKey(30))
    idx = jnp.arange(16)
    f, w, x = buffer['states'][idx], buffer['likelihood'][idx], buffer['x_true'][idx]
    key = jax.random.PRNGKey(31)
    tcfg.warmup_steps, tcfg.learning_rate, tcfg.num_iterations = 0, 1e-3, 100
    optimizer, opt_state = init_optimizer(model, tcfg)
    before = float(gold_loss(model, f, w, x, key)[0])
    m4, s4 = model, opt_state
    for _ in range(30):
        m4, s4, loss, _ = train_step(m4, optimizer, s4, f, w, x, key, 4)
        assert jnp.isfinite(loss)
    assert float(gold_loss(m4, f, w, x, key)[0]) < before
    a = train_step(model, optimizer, opt_state, f, w, x, key, 4)
    b = train_step(model, optimizer, opt_state, f, w, x, key, 4)
    assert float(a[2]) == float(b[2])
    with pytest.raises(ValueError):
        train_step(model, optimizer, opt_state, f, w, x, key, 3)


def test_train_step_reduces_gold_loss_on_fixed_batch(setup):
    cfg, env, model = setup
    tcfg = cfg.training
    buffer, _ = collect_rollout(env, model, tcfg, simulate_episodes, jax.random.PRNGKey(9))
    idx = jnp.arange(min(16, buffer['x_true'].shape[0]))
    f, w, x = buffer['states'][idx], buffer['likelihood'][idx], buffer['x_true'][idx]
    key = jax.random.PRNGKey(10)
    tcfg.warmup_steps, tcfg.learning_rate, tcfg.num_iterations = 0, 1e-3, 100
    optimizer, opt_state = init_optimizer(model, tcfg)
    before = float(gold_loss(model, f, w, x, key)[0])
    assert jnp.isfinite(before)
    for _ in range(30):
        model, opt_state, loss, _ = train_step(model, optimizer, opt_state, f, w, x, key)
    after = float(gold_loss(model, f, w, x, key)[0])
    assert after < before


def test_validate_reports_posterior_and_forecast_rmse(setup):
    cfg, env, model = setup
    tcfg = cfg.training
    tcfg.val_posterior_samples = 8
    val_eps = simulate_episodes(env, 2, jax.random.PRNGKey(30))
    rmodel = rollout_model_from(model, cfg.model, 3, False, jax.random.PRNGKey(31))
    v = validate(model, rmodel, val_eps, tcfg, jax.random.PRNGKey(32), 3.9)
    assert {'gold_loss', 'post_rmse', 'forecast_rmse', 'neff_over_n', 'resample_frac'} <= set(v)
    assert all(np.isfinite(x) for x in v.values())
    assert v['post_rmse'] > 0.0 and v['forecast_rmse'] > 0.0


@pytest.mark.slow
def test_domain_trains_end_to_end(domain, tmp_path):
    """Each domain's trainer runs its smoke config: rollouts, gradient steps, validation, logs
    and checkpoints."""

    cfg = domain.config([f'save_path={tmp_path}'])

    domain.module.run(cfg)

    with open(tmp_path / 'train_log.jsonl') as f:
        log = [json.loads(line) for line in f]
    assert log[-1]['step'] == cfg.training.num_iterations
    assert all(np.isfinite(record['loss']) for record in log)
    assert any('val' in record for record in log)
    saved = {p.name for p in tmp_path.iterdir()}
    assert {'model.eqx', 'model_config.json', 'data_config.json', 'train_config.json'} <= saved
