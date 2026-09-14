"""On-policy gold-state training for the Lorenz-96 NBF embedding.

No stored particle corpus. Episodes (true states and observations) are simulated on
the fly, and the weighted particle sets the encoder sees are produced along each
episode by a filter rollout. The loss at every step is the generator objective
(conditional flow matching for the FM generator, the exact NLL for a normalizing
flow) evaluated at the single true state under the embedding of that step's set.
Since (x_t, y_{1:t}) is a joint draw from the model, this is a proper scoring rule
whose minimizer is p(x_t | particle set).

Two rollout families produce the sets:

* `nbf` / `nbf_op` (default, on-policy): the NBF filter itself, with the current
  parameters, rolled from the start of each episode, produces the sets. Faithful to
  deployment, including the compounding of the model's own errors down the episode,
  at the price of a full filter rollout per cycle.
* `pf` / `pf_op` (model-free, cheap): a reference particle filter with
  `reference_num_particles` particles tracks the posterior along the episode. At each
  step, `rollout_num_particles` fresh draws from the step t-1 reference posterior are
  propagated and reweighted against y_t (blind proposal for `pf`, optimal proposal for
  `pf_op`), and resampled when ESS < n/2, exactly the set NBF / NBF-OP would embed with
  the reference posterior standing in for the generator. No network in the loop, but
  the sets carry a PF's error shape rather than NBF's own; a control arm, not the
  default.

`*_mixed` alternates the blind and optimal-proposal variants so one checkpoint is
trained for both eval rows. The rollout is treated as data: gradients flow through
`embed` and `generator.loss` for the step being scored, never through the rollout.

Design doc: agents/nbf/GOLD_STATE_TRAINING.md (polar-rl).
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from embedex.distributions.builder import create_model
from embedex.distributions.config import ModelConfig
from embedex.distributions.training import init_optimizer
from jax import Array

from nbf.config import apply_overrides, build_dataclass
from nbf.domains.lorenz96.config import DataConfig
from nbf.domains.lorenz96.config import TrainingConfig as _L96TrainingConfig
from nbf.domains.lorenz96.env import Lorenz96Env, build_lorenz96_env, rk4_window
from nbf.eval.base import cast_velocity_net_bf16, env_rollout_batch
from nbf.filters.nbf import NeuralBayesianFilter
from nbf.filters.nbf_op import OptimalProposalNeuralBayesianFilter
from nbf.filters.particle import systematic_indices
from nbf.train import _count_parameters

ROLLOUT_FILTERS = ('pf', 'pf_op', 'pf_mixed', 'nbf', 'nbf_op', 'mixed')
MODEL_FREE_KINDS = ('pf', 'pf_op')


def rollout_kinds(rollout_filter: str) -> tuple[str, ...]:
    if rollout_filter == 'pf_mixed':
        return ('pf', 'pf_op')
    if rollout_filter == 'mixed':
        return ('nbf', 'nbf_op')
    return (rollout_filter,)


@dataclass
class GoldTrainingConfig(_L96TrainingConfig):
    # `batch_size` is the number of (particle set, true state) rows per gradient step.
    # `num_samples` is unused here (the set size is `rollout_num_particles`); kept for
    # dataclass compatibility with the base config.
    rollout_filter: str = 'mixed'  # nbf | nbf_op | mixed (on-policy) | pf | pf_op | pf_mixed
    rollout_num_particles: int = 2048  # n at deployment; the encoder trains on exactly this
    reference_num_particles: int = 65536  # reference PF size for the model-free rollouts
    episodes_per_rollout: int = 64
    grad_steps_per_rollout: int = 20
    rollout_fm_steps: int = 0  # 0 = the model's own fm_num_steps (eval-faithful, slow)
    rollout_bf16: bool = False  # bf16 velocity net during rollouts only
    norm_episodes: int = 512  # episodes used to estimate the climatological norm stats
    val_episodes: int = 64
    val_interval: int = 5000


@dataclass
class GoldConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: GoldTrainingConfig = field(default_factory=GoldTrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    save_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_gold')
    seed: int = 0


def _src_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def load_config(path: str | None, overrides: list[str]) -> GoldConfig:
    if path is None:
        cfg = GoldConfig()
    else:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = build_dataclass(GoldConfig, raw)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    if cfg.training.rollout_filter not in ROLLOUT_FILTERS:
        raise ValueError(
            f'training.rollout_filter must be one of {ROLLOUT_FILTERS}, '
            f'got {cfg.training.rollout_filter!r}'
        )
    if not os.path.isabs(cfg.save_path):
        cfg.save_path = os.path.normpath(os.path.join(_src_root(), cfg.save_path))
    return cfg


# --------------------------------------------------------------------------------------
# Episodes and normalization
# --------------------------------------------------------------------------------------


def estimate_norm_stats(env: Lorenz96Env, num_episodes: int, key: Array) -> tuple[float, float]:
    """Climatological mean/std of the true state, the same statistic datagen used."""
    states_all, _ = env_rollout_batch(env, env.time_limit, jax.random.split(key, num_episodes))
    x = np.asarray(states_all.x, dtype=np.float64)
    return float(x.mean()), float(x.std())


@eqx.filter_jit
def simulate_episodes(env: Lorenz96Env, num_episodes: int, key: Array):
    """(state0, obs0, states_next, obs_prev, obs_next, x_all) for a batch of episodes.

    `x_all` is the raw true trajectory of shape (B, T + 1, K); everything else matches
    the eval streams' argument layout so the filter code path is shared.
    """
    states_all, obs_all = env_rollout_batch(
        env, env.time_limit, jax.random.split(key, num_episodes)
    )
    state0 = jax.tree.map(lambda x: x[:, 0], states_all)
    obs0 = jax.tree.map(lambda x: x[:, 0], obs_all)
    states_next = jax.tree.map(lambda x: x[:, 1:], states_all)
    obs_prev = jax.tree.map(lambda x: x[:, :-1], obs_all)
    obs_next = jax.tree.map(lambda x: x[:, 1:], obs_all)
    return state0, obs0, states_next, obs_prev, obs_next, states_all.x


# --------------------------------------------------------------------------------------
# On-policy rollouts
# --------------------------------------------------------------------------------------


def make_filter(model, kind: str, num_particles: int) -> NeuralBayesianFilter:
    if kind == 'nbf':
        return NeuralBayesianFilter(model, num_particles=num_particles)
    if kind == 'nbf_op':
        return OptimalProposalNeuralBayesianFilter(model, num_particles=num_particles)
    raise ValueError(f'unknown rollout filter {kind!r}')


def rollout_model_from(model, model_cfg: ModelConfig, fm_steps: int, bf16: bool, key: Array):
    """A copy of `model` for rollouts, optionally with fewer ODE steps and a bf16 velocity net.

    `num_steps` is a static field, so the cheap ODE variant is built by instantiating the
    generator with the modified config and swapping the trained velocity net into it.
    """
    if fm_steps and fm_steps != model_cfg.generator.fm_num_steps:
        import copy

        cfg = copy.deepcopy(model_cfg)
        cfg.generator.fm_num_steps = int(fm_steps)
        template = create_model(cfg, dequantize=False, key=key)
        # The velocity net carries every trained generator leaf; the template supplies the
        # static `num_steps`. The encoder is shared by reference.
        generator = eqx.tree_at(
            lambda g: g.velocity_net, template.generator, model.generator.velocity_net
        )
        model = eqx.tree_at(lambda m: m.generator, model, generator)
    if bf16:
        model = cast_velocity_net_bf16(model)
    return model


@eqx.filter_jit
def rollout_particle_sets(
    env: Lorenz96Env,
    filt: NeuralBayesianFilter,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    reset_keys: Array,
    step_keys_b: Array,
):
    """Run the filter along each episode and return every weighted set it embeds.

    Returns features (B, T + 1, n, K) in normalized coordinates, weights (B, T + 1, n),
    and neff (B, T) for the T update steps. Row t of the leading time axis is the set
    whose embedding represents the belief over x_t (row 0 is the prior set).
    """

    def _one(state0, obs0, states_next, obs_prev, obs_next, rkey, step_keys):
        feats0, w0 = filt.reset_particles(env, state0, obs0, rkey)
        emb0 = filt.model.embed(feats0, w0)[0]

        def _step(embedding, xs):
            sk, state_t, prev_obs_t, next_obs_t = xs
            pset, neff, _, _ = filt.update_particles(
                env, embedding, state_t, prev_obs_t, 0, next_obs_t, sk
            )
            feats = jax.vmap(lambda p: p.to_flat_features())(pset.particles)
            embedding = filt.model.embed(feats, pset.weights)[0]
            return embedding, (feats, pset.weights, neff)

        _, (feats, ws, neffs) = jax.lax.scan(
            _step, emb0, (step_keys, states_next, obs_prev, obs_next)
        )
        feats = jnp.concatenate([feats0[None], feats], axis=0)
        ws = jnp.concatenate([w0[None], ws], axis=0)
        return feats, ws, neffs

    return jax.vmap(_one)(
        state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b, reset_keys, step_keys_b
    )


@eqx.filter_jit
def pf_rollout_particle_sets(
    env: Lorenz96Env,
    kind: str,
    num_particles: int,
    reference_num_particles: int,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    reset_keys: Array,
    step_keys_b: Array,
):
    """Model-free stand-in for `rollout_particle_sets`: same output layout.

    A reference PF (N = `reference_num_particles`, same proposal as `kind`, systematic
    resample every step) tracks the posterior. The training set at step t is n fresh
    draws from the reference posterior at t-1, propagated and reweighted against y_t,
    resampled when ESS < n/2, which is the set NBF embeds with the reference posterior
    in place of the generator. Row 0 is the prior set, as in `reset_particles`.
    """
    optimal = kind == 'pf_op'
    n, N, K = num_particles, reference_num_particles, env.K
    obs_idx = env.obs_idx

    def _prior(state0, key, count):
        return state0.center + env.sigma_init * jax.random.normal(key, (count, K))

    def _move_and_weight(x_prev, forcing, obs, key):
        means = rk4_window(x_prev, forcing, env.dt, env.substeps, env.linear)
        if optimal:
            log_w = env.marginal_observation_log_prob(means, obs)
            moved = env.conjugate_move(means, obs, key)
        else:
            moved = means + env.sigma_dyn * jax.random.normal(key, means.shape)
            log_w = -0.5 * jnp.sum(((moved[:, obs_idx] - obs.values) / obs.sigma_obs) ** 2, -1)
        log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)
        return moved, jax.nn.softmax(log_w)

    def _one(state0, obs0, states_next, obs_prev, obs_next, rkey, step_keys):
        k_ref, k_row = jax.random.split(rkey)
        ref_x = _prior(state0, k_ref, N)
        ref_w = jnp.full(N, 1.0 / N)
        row0 = _prior(state0, k_row, n)
        w0 = jnp.full(n, 1.0 / n)
        forcing = state0.forcing

        def _step(carry, xs):
            ref_x, ref_w = carry
            sk, _state_t, _prev_obs, next_obs = xs
            k_draw, k_move, k_res, k_ref_move, k_ref_res = jax.random.split(sk, 5)
            # "generate": n fresh draws from the reference posterior at t-1
            idx = jax.random.choice(k_draw, N, (n,), replace=True, p=ref_w)
            moved, w = _move_and_weight(ref_x[idx], forcing, next_obs, k_move)
            neff = 1.0 / jnp.sum(w**2)
            res_idx = systematic_indices(w, k_res)
            moved, w = jax.lax.cond(
                neff < 0.5 * n,
                lambda: (moved[res_idx], jnp.full(n, 1.0 / n)),
                lambda: (moved, w),
            )
            # advance the reference posterior to t
            ref_moved, ref_wt = _move_and_weight(ref_x, forcing, next_obs, k_ref_move)
            ref_x = ref_moved[systematic_indices(ref_wt, k_ref_res)]
            ref_w = jnp.full(N, 1.0 / N)
            return (ref_x, ref_w), (moved, w, neff)

        _, (xs_t, ws, neffs) = jax.lax.scan(
            _step, (ref_x, ref_w), (step_keys, states_next, obs_prev, obs_next)
        )
        xs_all = jnp.concatenate([row0[None], xs_t], axis=0)
        feats = (xs_all - env.norm_mean) / env.norm_std
        ws = jnp.concatenate([w0[None], ws], axis=0)
        return feats, ws, neffs

    return jax.vmap(_one)(
        state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b, reset_keys, step_keys_b
    )


def rollout_sets(env, rollout_model, kind, cfg: GoldTrainingConfig, eps, reset_keys, step_keys):
    """Dispatch to the model-free or on-policy rollout; identical output layout."""
    state0, obs0, states_next, obs_prev, obs_next, _ = eps
    args = (state0, obs0, states_next, obs_prev, obs_next, reset_keys, step_keys)
    if kind in MODEL_FREE_KINDS:
        return pf_rollout_particle_sets(
            env, kind, cfg.rollout_num_particles, cfg.reference_num_particles, *args
        )
    filt = make_filter(rollout_model, kind, cfg.rollout_num_particles)
    return rollout_particle_sets(env, filt, *args)


def collect_rollout(env, rollout_model, kind, cfg: GoldTrainingConfig, key: Array):
    """Simulate a batch of episodes, roll a filter along them, flatten to training rows."""
    ep_key, reset_key, step_key = jax.random.split(key, 3)
    B, T = cfg.episodes_per_rollout, env.time_limit
    eps = simulate_episodes(env, B, ep_key)
    x_all = eps[-1]
    feats, ws, neffs = rollout_sets(
        env,
        rollout_model,
        kind,
        cfg,
        eps,
        jax.random.split(reset_key, B),
        jax.random.split(step_key, B * T).reshape(B, T, -1),
    )
    x_norm = (x_all - env.norm_mean) / env.norm_std
    rows = B * (T + 1)
    n, K = feats.shape[-2], feats.shape[-1]
    buffer = {
        'features': feats.reshape(rows, n, K),
        'weights': ws.reshape(rows, n),
        'x_true': x_norm.reshape(rows, K),
    }
    stats = {
        'neff_over_n': float(jnp.mean(neffs)) / cfg.rollout_num_particles,
        'resample_frac': float(jnp.mean(neffs < 0.5 * cfg.rollout_num_particles)),
    }
    return buffer, stats


# --------------------------------------------------------------------------------------
# Loss and optimization
# --------------------------------------------------------------------------------------


def gold_loss(model, features: Array, weights: Array, x_true: Array, key: Array):
    """Mean generator loss of the true states under the embeddings of their sets."""
    zs, norms = jax.vmap(model.embed)(features, weights)
    keys = jax.random.split(key, x_true.shape[0])
    losses, _, _ = jax.vmap(model.generator.loss, in_axes=(0, 0, None, 0))(x_true, zs, None, keys)
    return jnp.mean(losses), jnp.mean(norms)


@eqx.filter_jit
def train_step(model, optimizer, opt_state, features, weights, x_true, key):
    (loss, mean_norm), grads = eqx.filter_value_and_grad(gold_loss, has_aux=True)(
        model, features, weights, x_true, key
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, mean_norm


@eqx.filter_jit
def eval_loss(model, features, weights, x_true, key):
    return gold_loss(model, features, weights, x_true, key)


@eqx.filter_jit
def weighted_mean_rmse(features, weights, x_true, norm_std):
    """RMSE of the weighted particle mean against the truth, in raw state units."""
    w = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-30)
    mean = jnp.sum(features * w[..., None], axis=-2)
    return jnp.sqrt(jnp.mean((mean - x_true) ** 2)) * norm_std


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------


def _save_checkpoint(save_path, model, opt_state, step, numbered=False):
    with open(os.path.join(save_path, 'model.eqx'), 'wb') as f:
        eqx.tree_serialise_leaves(f, model)
    with open(os.path.join(save_path, 'opt_state.eqx'), 'wb') as f:
        eqx.tree_serialise_leaves(f, opt_state)
    with open(os.path.join(save_path, 'train_state.json'), 'w') as f:
        json.dump({'step': int(step)}, f)
    if numbered:
        with open(os.path.join(save_path, f'model_{step:06d}.eqx'), 'wb') as f:
            eqx.tree_serialise_leaves(f, model)


def _try_resume(save_path, model, opt_state):
    state_path = os.path.join(save_path, 'train_state.json')
    if not os.path.exists(state_path):
        return model, opt_state, 0
    with open(state_path) as f:
        step = int(json.load(f)['step'])
    with open(os.path.join(save_path, 'model.eqx'), 'rb') as f:
        model = eqx.tree_deserialise_leaves(f, model)
    with open(os.path.join(save_path, 'opt_state.eqx'), 'rb') as f:
        opt_state = eqx.tree_deserialise_leaves(f, opt_state)
    print(f'[resume] step {step} from {save_path} (optimizer state and LR schedule restored)')
    return model, opt_state, step


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------


def run(cfg: GoldConfig) -> None:
    tcfg = cfg.training
    os.makedirs(cfg.save_path, exist_ok=True)
    key = jax.random.PRNGKey(cfg.seed)
    key, norm_key, model_key, val_key, roll_key = jax.random.split(key, 5)

    # Norm stats first (datagen wrote norm_stats.json; here they ride in data_config.json).
    if cfg.data.norm_mean is None or cfg.data.norm_std is None:
        env0 = build_lorenz96_env(cfg.data)
        cfg.data.norm_mean, cfg.data.norm_std = estimate_norm_stats(
            env0, tcfg.norm_episodes, norm_key
        )
    env = build_lorenz96_env(cfg.data)
    print(
        f'Climatological normalization: mean={cfg.data.norm_mean:.4f} '
        f'std={cfg.data.norm_std:.4f} (from {tcfg.norm_episodes} episodes)'
    )

    cfg.model.input_dim = cfg.model.output_dim = int(cfg.data.K)
    model = create_model(cfg.model, dequantize=False, key=model_key)
    print(f'Model parameters: {_count_parameters(model):,}')

    configs = {'model_config': cfg.model, 'data_config': cfg.data, 'train_config': tcfg}
    for name, obj in configs.items():
        with open(os.path.join(cfg.save_path, f'{name}.json'), 'w') as f:
            json.dump(asdict(obj), f, indent=2)

    if tcfg.warmup_steps >= tcfg.num_iterations:
        # An ITERS override on a smoke run would otherwise hand optax a negative cosine
        # decay; a short run gets a proportionally short warmup instead.
        clamped = max(1, tcfg.num_iterations // 10)
        print(f'[warmup] warmup_steps {tcfg.warmup_steps} >= num_iterations, using {clamped}')
        tcfg.warmup_steps = clamped
    optimizer, opt_state = init_optimizer(model, tcfg)
    model, opt_state, step = _try_resume(cfg.save_path, model, opt_state)

    # Fixed validation episodes, rolled out fresh with the current model at each eval.
    val_env = env
    val_eps = simulate_episodes(val_env, tcfg.val_episodes, val_key)

    kinds = rollout_kinds(tcfg.rollout_filter)
    log_path = os.path.join(cfg.save_path, 'train_log.jsonl')
    rollout_idx = step // tcfg.grad_steps_per_rollout
    t_start = time.time()

    while step < tcfg.num_iterations:
        kind = kinds[rollout_idx % len(kinds)]
        # Keyed by rollout index so a resumed run replays the same episode stream.
        rk, tk, vk = jax.random.split(jax.random.fold_in(roll_key, rollout_idx), 3)
        rmodel = (
            None
            if kind in MODEL_FREE_KINDS
            else rollout_model_from(model, cfg.model, tcfg.rollout_fm_steps, tcfg.rollout_bf16, rk)
        )

        t0 = time.time()
        buffer, rstats = collect_rollout(env, rmodel, kind, tcfg, rk)
        t_roll = time.time() - t0
        rows = buffer['x_true'].shape[0]

        t0 = time.time()
        losses, norms = [], []
        for _ in range(tcfg.grad_steps_per_rollout):
            if step >= tcfg.num_iterations:
                break
            tk, idx_key, step_key = jax.random.split(tk, 3)
            idx = jax.random.randint(idx_key, (tcfg.batch_size,), 0, rows)
            model, opt_state, loss, mean_norm = train_step(
                model,
                optimizer,
                opt_state,
                buffer['features'][idx],
                buffer['weights'][idx],
                buffer['x_true'][idx],
                step_key,
            )
            losses.append(float(loss))
            norms.append(float(mean_norm))
            step += 1
            if tcfg.checkpoint_interval > 0 and step % tcfg.checkpoint_interval == 0:
                _save_checkpoint(cfg.save_path, model, opt_state, step, numbered=True)
                print(f'[checkpoint @ step {step}] saved to {cfg.save_path}')
        t_train = time.time() - t0
        rollout_idx += 1

        record = {
            'step': step,
            'rollout': rollout_idx,
            'filter': kind,
            'loss': float(np.mean(losses)),
            'emb_norm': float(np.mean(norms)),
            'rollout_s': round(t_roll, 2),
            'train_s': round(t_train, 2),
            'elapsed_h': round((time.time() - t_start) / 3600, 3),
            **rstats,
        }

        if tcfg.val_interval > 0 and (
            step % tcfg.val_interval < tcfg.grad_steps_per_rollout or step >= tcfg.num_iterations
        ):
            vmodel = rmodel
            if vmodel is None and any(k not in MODEL_FREE_KINDS for k in kinds):
                vmodel = rollout_model_from(
                    model, cfg.model, tcfg.rollout_fm_steps, tcfg.rollout_bf16, rk
                )
            record['val'] = validate(val_env, model, vmodel, val_eps, kinds, tcfg, vk)

        with open(log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        every = max(1, tcfg.log_interval // tcfg.grad_steps_per_rollout)
        if rollout_idx % every == 0 or 'val' in record:
            print(json.dumps(record))

    _save_checkpoint(cfg.save_path, model, opt_state, step, numbered=False)
    print(f'Model saved to {os.path.join(cfg.save_path, "model.eqx")}')


def validate(env, model, rollout_model, val_eps, kinds, tcfg, key):
    """Gold loss and weighted-mean RMSE on the fixed validation episodes, per filter kind."""
    state0, obs0, states_next, obs_prev, obs_next, x_all = val_eps
    B, T = x_all.shape[0], env.time_limit
    x_norm = ((x_all - env.norm_mean) / env.norm_std).reshape(B * (T + 1), -1)
    out = {}
    for i, kind in enumerate(kinds):
        k = jax.random.fold_in(key, 1000 + i)
        rk, sk, lk = jax.random.split(k, 3)
        feats, ws, neffs = rollout_sets(
            env,
            rollout_model,
            kind,
            tcfg,
            val_eps,
            jax.random.split(rk, B),
            jax.random.split(sk, B * T).reshape(B, T, -1),
        )
        n, K = feats.shape[-2], feats.shape[-1]
        feats = feats.reshape(-1, n, K)
        ws = ws.reshape(-1, n)
        # Chunk the loss to bound memory; each chunk is a jitted call of fixed shape.
        chunk = tcfg.batch_size
        rows = feats.shape[0]
        losses = []
        for s in range(0, rows - rows % chunk, chunk):
            lk, ck = jax.random.split(lk)
            sl = slice(s, s + chunk)
            losses.append(float(eval_loss(model, feats[sl], ws[sl], x_norm[sl], ck)[0]))
        out[kind] = {
            'gold_loss': float(np.mean(losses)),
            'wmean_rmse': float(weighted_mean_rmse(feats, ws, x_norm, env.norm_std)),
            'neff_over_n': float(jnp.mean(neffs)) / tcfg.rollout_num_particles,
            'resample_frac': float(jnp.mean(neffs < 0.5 * tcfg.rollout_num_particles)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='On-policy gold-state training for L96 NBF.')
    parser.add_argument('--config', type=str, default=None, help='YAML config path')
    parser.add_argument(
        '--set', nargs='*', default=[], help='Override config values: key.path=value'
    )
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    print(json.dumps({'save_path': cfg.save_path, 'training': asdict(cfg.training)}, indent=1))
    run(cfg)


if __name__ == '__main__':
    main()
