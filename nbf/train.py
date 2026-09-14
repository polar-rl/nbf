"""On-policy training of an NBF embedding, shared by every domain.

No stored particle corpus and no reference filter. Episodes (true states and
observations) are simulated on the fly, and the particle sets the encoder sees are
produced along each episode by the NBF filter itself, rolled out with the current
parameters. A set is the forecast particles together with their likelihood features
(`nbf.filters.likelihood`), uniformly weighted. The loss at every step is the generator objective (conditional
flow matching for the FM generator, the exact NLL for a normalizing flow or
autoregressive generator) evaluated at the single true state under the embedding of that
step's set. Since (x_t, y_{1:t}) is a joint draw from the model, this is a proper scoring
rule whose population minimizer is p(x_t | set), the filtering posterior in the limit of
the set.

The rollout is treated as data: gradients flow through `embed` and `generator.loss`
for the step being scored, never through the rollout. A domain supplies an env, a
`simulate_episodes(env, num_episodes, key) -> Episodes` function and a `Config`;
everything else lives here. See `nbf.domains.lorenz96.train` for the pattern.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from embedex.distributions.builder import create_model
from embedex.distributions.config import ModelConfig
from embedex.distributions.training import init_optimizer
from jax import Array

from nbf.config import TrainingConfig as _BaseTrainingConfig
from nbf.eval.base import cast_velocity_net_bf16
from nbf.filters.nbf import NeuralBayesianFilter


def _count_parameters(model):
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return sum(leaf.size for leaf in leaves)


@dataclass
class OnPolicyTrainingConfig(_BaseTrainingConfig):
    # `batch_size` is the number of (particle set, true state) rows per gradient step.
    rollout_num_particles: int = 2048  # n at deployment; the encoder trains on exactly this
    episodes_per_rollout: int = 64
    grad_steps_per_rollout: int = 20
    rollout_fm_steps: int = 0  # 0 = the model's own fm_num_steps (eval-faithful, slow)
    rollout_bf16: bool = False  # bf16 velocity net during rollouts only
    micro_batches: int = 1  # gradient accumulation: batch_size rows in this many chunks
    val_episodes: int = 64
    val_interval: int = 5000
    val_posterior_samples: int = 64  # generator samples per set for the validation RMSE
    resume_interval_s: float = 1200.0  # rewrite the resume checkpoint this often (wall clock)


class Episodes(NamedTuple):
    """A batch of simulated episodes in the layout the filter streams consume.

    `x_norm` is the true trajectory in the model's normalized coordinates, shape
    (B, T + 1, D). `env` is the environment to roll the filter in and `env_axes` the
    vmap in_axes tree for it: None when one env serves every episode (Lorenz-96),
    or a tree with 0 on the per-episode leaves (pursuit-evasion policies).
    """

    state0: object
    obs0: object
    states_next: object
    obs_prev: object
    obs_next: object
    x_norm: Array
    env: object
    env_axes: object = None


# --------------------------------------------------------------------------------------
# On-policy rollouts
# --------------------------------------------------------------------------------------


def make_filter(model, num_particles: int) -> NeuralBayesianFilter:
    return NeuralBayesianFilter(model, num_particles)


_ROLLOUT_TEMPLATES: dict = {}


def _rollout_generator_template(model_cfg: ModelConfig, fm_steps: int, key: Array):
    """The generator skeleton with `num_steps = fm_steps`, built once per config.

    Rebuilding it every rollout gave `rollout_particle_sets` a fresh set of static leaves
    each time, so the jitted rollout was retraced and recompiled on every call and the jit
    cache grew without bound (host OOM after ~450 rollouts at K=40).
    """
    ck = (int(fm_steps), json.dumps(asdict(model_cfg), sort_keys=True, default=str))
    if ck not in _ROLLOUT_TEMPLATES:
        import copy

        cfg = copy.deepcopy(model_cfg)
        cfg.generator.fm_num_steps = int(fm_steps)
        _ROLLOUT_TEMPLATES[ck] = create_model(cfg, dequantize=False, key=key).generator
    return _ROLLOUT_TEMPLATES[ck]


def rollout_model_from(model, model_cfg: ModelConfig, fm_steps: int, bf16: bool, key: Array):
    """A copy of `model` for rollouts, optionally with fewer ODE steps and a bf16 velocity net.

    `num_steps` is a static field, so the cheap ODE variant is the cached generator template
    with the trained velocity net swapped in. The encoder is shared by reference.
    """
    if fm_steps and fm_steps != model_cfg.generator.fm_num_steps:
        template = _rollout_generator_template(model_cfg, fm_steps, key)
        generator = eqx.tree_at(lambda g: g.velocity_net, template, model.generator.velocity_net)
        model = eqx.tree_at(lambda m: m.generator, model, generator)
    if bf16:
        model = cast_velocity_net_bf16(model)
    return model


@eqx.filter_jit
def rollout_particle_sets(
    env,
    filt: NeuralBayesianFilter,
    state0_b,
    obs0_b,
    states_next_b,
    obs_prev_b,
    obs_next_b,
    reset_keys: Array,
    step_keys_b: Array,
    env_axes=None,
):
    """Run the filter along each episode and return every weighted set it embeds.

    Returns the encoder inputs, normalized states (B, T + 1, n, D) and likelihood features
    (B, T + 1, n, d), and neff (B, T) for the T update steps.
    Row t of the leading time axis is the set whose embedding represents the belief over
    x_t (row 0 is the prior set).
    """

    def _one(env, state0, obs0, states_next, obs_prev, obs_next, rkey, step_keys):
        states0, lik0 = filt.reset_particles(env, state0, obs0, rkey)
        emb0 = filt.model.embed(states0, lik0)[0]

        def _step(embedding, xs):
            sk, state_t, prev_obs_t, next_obs_t = xs
            pset, neff, _, _ = filt.update_particles(
                env, embedding, state_t, prev_obs_t, 0, next_obs_t, sk
            )
            states, lik = filt.encoder_input(pset)
            embedding = filt.model.embed(states, lik)[0]
            return embedding, (states, lik, neff)

        _, (states, lik, neffs) = jax.lax.scan(
            _step, emb0, (step_keys, states_next, obs_prev, obs_next)
        )
        states = jnp.concatenate([states0[None], states], axis=0)
        lik = jnp.concatenate([lik0[None], lik], axis=0)
        return states, lik, neffs

    return jax.vmap(_one, in_axes=(env_axes, 0, 0, 0, 0, 0, 0, 0))(
        env, state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b, reset_keys, step_keys_b
    )


def rollout_sets(rollout_model, cfg, eps: Episodes, reset_keys, step_keys):
    filt = make_filter(rollout_model, cfg.rollout_num_particles)
    return rollout_particle_sets(
        eps.env,
        filt,
        eps.state0,
        eps.obs0,
        eps.states_next,
        eps.obs_prev,
        eps.obs_next,
        reset_keys,
        step_keys,
        env_axes=eps.env_axes,
    )


def collect_rollout(env, rollout_model, cfg, simulate_episodes, key: Array):
    """Simulate a batch of episodes, roll the filter along them and flatten to training
    rows."""
    ep_key, reset_key, step_key = jax.random.split(key, 3)
    B, T = cfg.episodes_per_rollout, env.time_limit
    eps = simulate_episodes(env, B, ep_key)
    reset_keys = jax.random.split(reset_key, B)
    step_keys = jax.random.split(step_key, B * T).reshape(B, T, -1)
    states, lik, neffs = rollout_sets(rollout_model, cfg, eps, reset_keys, step_keys)
    rows = B * (T + 1)
    n, D, d = states.shape[-2], states.shape[-1], lik.shape[-1]
    buffer = {
        'states': states.reshape(rows, n, D),
        'likelihood': lik.reshape(rows, n, d),
        'x_true': eps.x_norm.reshape(rows, eps.x_norm.shape[-1]),
    }
    stats = {
        'neff_over_n': float(jnp.mean(neffs)) / cfg.rollout_num_particles,
        'resample_frac': float(jnp.mean(neffs < 0.5 * cfg.rollout_num_particles)),
    }
    return buffer, stats


# --------------------------------------------------------------------------------------
# Loss and optimization
# --------------------------------------------------------------------------------------


def gold_loss(model, states: Array, likelihood: Array, x_true: Array, key: Array):
    """Mean generator loss of the true states under the embeddings of their sets."""
    zs, norms = jax.vmap(model.embed)(states, likelihood)
    keys = jax.random.split(key, x_true.shape[0])
    losses, _, _ = jax.vmap(model.generator.loss, in_axes=(0, 0, None, 0))(x_true, zs, None, keys)
    return jnp.mean(losses), jnp.mean(norms)


@eqx.filter_jit
def train_step(model, optimizer, opt_state, states, likelihood, x_true, key, micro_batches=1):
    """One optimizer step on `batch_size` rows, with the gradient accumulated over
    `micro_batches` equal chunks so activation memory scales with batch_size /
    micro_batches (every row's set is embedded in full, so the encoder's activations
    are what bound memory at large n and K)."""
    M = int(micro_batches)
    if M == 1:
        (loss, mean_norm), grads = eqx.filter_value_and_grad(gold_loss, has_aux=True)(
            model, states, likelihood, x_true, key
        )
    else:
        rows = states.shape[0]
        if rows % M:
            raise ValueError(f'batch_size {rows} must be a multiple of micro_batches {M}')

        def chunks(x):
            return x.reshape((M, rows // M) + x.shape[1:])

        params = eqx.filter(model, eqx.is_inexact_array)

        def _accumulate(carry, xs):
            f, w, x, k = xs
            (li, nrm), g = eqx.filter_value_and_grad(gold_loss, has_aux=True)(model, f, w, x, k)
            acc, l_acc, n_acc = carry
            acc = jax.tree_util.tree_map(lambda a, b: a + b, acc, g)
            return (acc, l_acc + li, n_acc + nrm), None

        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        (grads, loss, mean_norm), _ = jax.lax.scan(
            _accumulate,
            (zeros, jnp.zeros(()), jnp.zeros(())),
            (chunks(states), chunks(likelihood), chunks(x_true), jax.random.split(key, M)),
        )
        grads = jax.tree_util.tree_map(lambda g: g / M, grads)
        loss, mean_norm = loss / M, mean_norm / M
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, mean_norm


@eqx.filter_jit
def eval_loss(model, states, likelihood, x_true, key):
    return gold_loss(model, states, likelihood, x_true, key)


@eqx.filter_jit
def forecast_mean_rmse(states, x_true, scale=1.0):
    """RMSE of the sets' mean state (the forecast mean under uniform weights) against the
    truth; `scale` maps normalized coordinates back to state units."""
    mean = jnp.mean(states, axis=-2)
    return jnp.sqrt(jnp.mean((mean - x_true) ** 2)) * scale


@eqx.filter_jit
def posterior_mean_sq_error(model, states, likelihood, x_true, num_samples, key):
    """Per-row squared error of the posterior mean (generator samples under each set's
    embedding) against the truth, in normalized coordinates; the eval grid's RMSE."""
    zs, _ = jax.vmap(model.embed)(states, likelihood)
    keys = jax.random.split(key, zs.shape[0])
    samples = jax.vmap(lambda z, k: model.generate(z, num_samples, key=k))(zs, keys)
    return jnp.mean((jnp.mean(samples, axis=1) - x_true) ** 2, axis=-1)


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------


def _rss_gb() -> float:
    try:
        with open('/proc/self/statm') as f:
            return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1e9
    except (OSError, ValueError):
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _jit_cache_size(fn) -> int:
    try:
        return int(fn._cached._cache_size())
    except AttributeError:
        return -1


def save_checkpoint(save_path, model, opt_state, step, numbered=False):
    with open(os.path.join(save_path, 'model.eqx'), 'wb') as f:
        eqx.tree_serialise_leaves(f, model)
    with open(os.path.join(save_path, 'opt_state.eqx'), 'wb') as f:
        eqx.tree_serialise_leaves(f, opt_state)
    with open(os.path.join(save_path, 'train_state.json'), 'w') as f:
        json.dump({'step': int(step)}, f)
    if numbered:
        with open(os.path.join(save_path, f'model_{step:06d}.eqx'), 'wb') as f:
            eqx.tree_serialise_leaves(f, model)


def try_resume(save_path, model, opt_state):
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


def run(cfg, *, env, simulate_episodes, rmse_scale: float = 1.0) -> None:
    """Train `cfg.model` on-policy in `env`.

    `cfg` has `model`, `training` (OnPolicyTrainingConfig), `data`, `save_path`, `seed`.
    `simulate_episodes(env, num_episodes, key)` returns an `Episodes` batch.
    """
    tcfg = cfg.training
    os.makedirs(cfg.save_path, exist_ok=True)
    key = jax.random.PRNGKey(cfg.seed)
    key, model_key, val_key, roll_key = jax.random.split(key, 4)

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
    model, opt_state, step = try_resume(cfg.save_path, model, opt_state)

    # Fixed validation episodes, rolled out fresh with the current model at each eval.
    val_eps = simulate_episodes(env, tcfg.val_episodes, val_key)

    log_path = os.path.join(cfg.save_path, 'train_log.jsonl')
    rollout_idx = step // tcfg.grad_steps_per_rollout
    t_start = time.time()
    t_saved = time.time()
    warned_retrace = False

    while step < tcfg.num_iterations:
        # Keyed by rollout index so a resumed run replays the same episode stream.
        rk, tk, vk = jax.random.split(jax.random.fold_in(roll_key, rollout_idx), 3)
        rmodel = rollout_model_from(model, cfg.model, tcfg.rollout_fm_steps, tcfg.rollout_bf16, rk)

        t0 = time.time()
        buffer, rstats = collect_rollout(env, rmodel, tcfg, simulate_episodes, rk)
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
                buffer['states'][idx],
                buffer['likelihood'][idx],
                buffer['x_true'][idx],
                step_key,
                tcfg.micro_batches,
            )
            losses.append(float(loss))
            norms.append(float(mean_norm))
            step += 1
            if tcfg.checkpoint_interval > 0 and step % tcfg.checkpoint_interval == 0:
                save_checkpoint(cfg.save_path, model, opt_state, step, numbered=True)
                print(f'[checkpoint @ step {step}] saved to {cfg.save_path}')
        t_train = time.time() - t0
        rollout_idx += 1

        if tcfg.resume_interval_s > 0 and time.time() - t_saved > tcfg.resume_interval_s:
            save_checkpoint(cfg.save_path, model, opt_state, step, numbered=False)
            t_saved = time.time()

        record = {
            'step': step,
            'rollout': rollout_idx,
            'loss': float(np.mean(losses)),
            'emb_norm': float(np.mean(norms)),
            'rollout_s': round(t_roll, 2),
            'train_s': round(t_train, 2),
            'elapsed_h': round((time.time() - t_start) / 3600, 3),
            'rss_gb': round(_rss_gb(), 2),
            'rollout_jit_cache': _jit_cache_size(rollout_particle_sets),
            **rstats,
        }
        if record['rollout_jit_cache'] > 4 and not warned_retrace:
            print(
                '[warn] rollout_particle_sets is being retraced every rollout; host memory will grow'
            )
            warned_retrace = True

        if tcfg.val_interval > 0 and (
            step % tcfg.val_interval < tcfg.grad_steps_per_rollout or step >= tcfg.num_iterations
        ):
            record['val'] = validate(model, rmodel, val_eps, tcfg, vk, rmse_scale)

        with open(log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        every = max(1, tcfg.log_interval // tcfg.grad_steps_per_rollout)
        if rollout_idx % every == 0 or 'val' in record:
            print(json.dumps(record))

    save_checkpoint(cfg.save_path, model, opt_state, step, numbered=False)
    print(f'Model saved to {os.path.join(cfg.save_path, "model.eqx")}')


def validate(model, rollout_model, val_eps: Episodes, tcfg, key, rmse_scale=1.0):
    """Gold loss, posterior-mean RMSE (generator samples, the eval grid's quantity) and
    forecast-mean RMSE on the fixed validation episodes, in state units."""
    x_all = val_eps.x_norm
    B, T = x_all.shape[0], x_all.shape[1] - 1
    x_norm = x_all.reshape(B * (T + 1), -1)
    rk, sk, lk = jax.random.split(key, 3)
    states, lik, neffs = rollout_sets(
        rollout_model,
        tcfg,
        val_eps,
        jax.random.split(rk, B),
        jax.random.split(sk, B * T).reshape(B, T, -1),
    )
    n, D, d = states.shape[-2], states.shape[-1], lik.shape[-1]
    states = states.reshape(-1, n, D)
    lik = lik.reshape(-1, n, d)
    # Chunk the loss to bound memory; each chunk is a jitted call of fixed shape.
    rows = states.shape[0]
    chunk = max(1, min(tcfg.batch_size // tcfg.micro_batches, rows))
    losses, sq_errors = [], []
    for s in range(0, rows - rows % chunk, chunk):
        lk, ck, gk = jax.random.split(lk, 3)
        sl = slice(s, s + chunk)
        losses.append(float(eval_loss(model, states[sl], lik[sl], x_norm[sl], ck)[0]))
        sq_errors.append(
            np.asarray(
                posterior_mean_sq_error(
                    model, states[sl], lik[sl], x_norm[sl], tcfg.val_posterior_samples, gk
                )
            )
        )
    return {
        'gold_loss': float(np.mean(losses)),
        'post_rmse': float(np.sqrt(np.mean(np.concatenate(sq_errors)))) * float(rmse_scale),
        'forecast_rmse': float(forecast_mean_rmse(states, x_norm, rmse_scale)),
        'neff_over_n': float(jnp.mean(neffs)) / tcfg.rollout_num_particles,
        'resample_frac': float(jnp.mean(neffs < 0.5 * tcfg.rollout_num_particles)),
    }
