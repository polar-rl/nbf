"""On-policy training of an NBF embedding, shared by every domain.

No stored particle corpus and no reference filter. Episodes (true states and
observations) are simulated on the fly, and the weighted particle sets the encoder
sees are produced along each episode by the NBF filter itself, rolled out with the
current parameters; each rollout's episodes are split across the configured proposals. The loss at every step is the
generator objective (conditional flow matching for the FM generator, the exact NLL
for a normalizing flow or autoregressive generator) evaluated at the single true
state under the embedding of that step's set. Since (x_t, y_{1:t}) is a joint draw
from the model, this is a proper scoring rule whose population minimizer is
p(x_t | particle set).

The rollout is treated as data: gradients flow through `embed` and `generator.loss`
for the step being scored, never through the rollout. A domain supplies an env, a
`simulate_episodes(env, num_episodes, key) -> Episodes` function and a `Config`;
everything else lives here. See `nbf.domains.lorenz96.train` for the pattern.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
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
from nbf.filters.proposals import PROPOSALS, make_proposal


def _count_parameters(model):
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return sum(leaf.size for leaf in leaves)


@dataclass
class OnPolicyTrainingConfig(_BaseTrainingConfig):
    # `batch_size` is the number of (particle set, true state) rows per gradient step.
    rollout_proposals: list = field(default_factory=lambda: ['bootstrap'])
    rollout_num_particles: int = 2048  # n at deployment; the encoder trains on exactly this
    episodes_per_rollout: int = 64
    grad_steps_per_rollout: int = 20
    rollout_fm_steps: int = 0  # 0 = the model's own fm_num_steps (eval-faithful, slow)
    rollout_bf16: bool = False  # bf16 velocity net during rollouts only
    micro_batches: int = 1  # gradient accumulation: batch_size rows in this many chunks
    val_episodes: int = 64
    val_interval: int = 5000


def normalize_proposals(cfg: OnPolicyTrainingConfig) -> None:
    """Accept `a,b` from a --set override as well as a YAML list; validate the names."""
    props = cfg.rollout_proposals
    if isinstance(props, str):
        props = [props]
    # apply_overrides hands a `--set training.rollout_proposals=a,b` through as ['a,b'].
    props = [q.strip() for p in props for q in str(p).split(',') if q.strip()]
    unknown = [p for p in props if p not in PROPOSALS]
    if not props or unknown:
        raise ValueError(
            f'training.rollout_proposals must be a non-empty subset of {sorted(PROPOSALS)}, '
            f'got {cfg.rollout_proposals!r}'
        )
    cfg.rollout_proposals = props


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


def make_filter(model, proposal: str, num_particles: int) -> NeuralBayesianFilter:
    return NeuralBayesianFilter(model, num_particles, proposal=make_proposal(proposal))


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

    Returns features (B, T + 1, n, D) in normalized coordinates, weights (B, T + 1, n),
    and neff (B, T) for the T update steps. Row t of the leading time axis is the set
    whose embedding represents the belief over x_t (row 0 is the prior set).
    """

    def _one(env, state0, obs0, states_next, obs_prev, obs_next, rkey, step_keys):
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

    return jax.vmap(_one, in_axes=(env_axes, 0, 0, 0, 0, 0, 0, 0))(
        env, state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b, reset_keys, step_keys_b
    )


def rollout_sets(rollout_model, proposal, cfg, eps: Episodes, reset_keys, step_keys):
    filt = make_filter(rollout_model, proposal, cfg.rollout_num_particles)
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


def _slice_episodes(eps: Episodes, start: int, stop: int) -> Episodes:
    """Episodes [start, stop) of a batch; per-episode env leaves are sliced with them."""
    take = lambda x: x[start:stop]  # noqa: E731
    env = eps.env
    if eps.env_axes is not None:
        env = jax.tree_util.tree_map(
            lambda x, ax: take(x) if ax == 0 else x, eps.env, eps.env_axes,
            is_leaf=lambda x: x is None,
        )
    return Episodes(
        *(jax.tree.map(take, t) for t in (eps.state0, eps.obs0, eps.states_next, eps.obs_prev, eps.obs_next)),
        take(eps.x_norm),
        env,
        eps.env_axes,
    )


def collect_rollout(env, rollout_model, proposals, cfg, simulate_episodes, key: Array):
    """Simulate a batch of episodes, roll the filter along them under each proposal, and
    flatten to training rows.

    The episodes are split evenly across `proposals` (one string is one proposal), so
    every gradient step draws from every proposal's set distribution and there is no
    rollout-to-rollout alternation. Each proposal gets its own episodes; the rollout
    cost is the same as a single-proposal rollout of the same batch.
    """
    if isinstance(proposals, str):
        proposals = [proposals]
    ep_key, reset_key, step_key = jax.random.split(key, 3)
    B, T = cfg.episodes_per_rollout, env.time_limit
    P = len(proposals)
    if B % P:
        raise ValueError(f'episodes_per_rollout {B} must be a multiple of {P} proposals')
    eps = simulate_episodes(env, B, ep_key)
    reset_keys = jax.random.split(reset_key, B)
    step_keys = jax.random.split(step_key, B * T).reshape(B, T, -1)
    parts = []
    for i, proposal in enumerate(proposals):
        lo, hi = i * B // P, (i + 1) * B // P
        parts.append(
            rollout_sets(rollout_model, proposal, cfg, _slice_episodes(eps, lo, hi),
                         reset_keys[lo:hi], step_keys[lo:hi])
        )
    feats, ws, neffs = (jnp.concatenate([p[j] for p in parts], axis=0) for j in range(3))
    rows = B * (T + 1)
    n, D = feats.shape[-2], feats.shape[-1]
    buffer = {
        'features': feats.reshape(rows, n, D),
        'weights': ws.reshape(rows, n),
        'x_true': eps.x_norm.reshape(rows, D),
    }
    stats = {
        'neff_over_n': float(jnp.mean(neffs)) / cfg.rollout_num_particles,
        'resample_frac': float(jnp.mean(neffs < 0.5 * cfg.rollout_num_particles)),
    }
    for proposal, (_, _, ne) in zip(proposals, parts, strict=True):
        stats[f'neff_over_n_{proposal}'] = float(jnp.mean(ne)) / cfg.rollout_num_particles
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
def train_step(model, optimizer, opt_state, features, weights, x_true, key, micro_batches=1):
    """One optimizer step on `batch_size` rows, with the gradient accumulated over
    `micro_batches` equal chunks so activation memory scales with batch_size /
    micro_batches (every row's set is embedded in full, so the encoder's activations
    are what bound memory at large n and K)."""
    M = int(micro_batches)
    if M == 1:
        (loss, mean_norm), grads = eqx.filter_value_and_grad(gold_loss, has_aux=True)(
            model, features, weights, x_true, key
        )
    else:
        rows = features.shape[0]
        if rows % M:
            raise ValueError(f'batch_size {rows} must be a multiple of micro_batches {M}')

        def chunks(x):
            return x.reshape((M, rows // M) + x.shape[1:])

        params = eqx.filter(model, eqx.is_inexact_array)

        def _accumulate(carry, xs):
            f, w, x, k = xs
            (l, nrm), g = eqx.filter_value_and_grad(gold_loss, has_aux=True)(model, f, w, x, k)
            acc, l_acc, n_acc = carry
            acc = jax.tree_util.tree_map(lambda a, b: a + b, acc, g)
            return (acc, l_acc + l, n_acc + nrm), None

        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        (grads, loss, mean_norm), _ = jax.lax.scan(
            _accumulate,
            (zeros, jnp.zeros(()), jnp.zeros(())),
            (chunks(features), chunks(weights), chunks(x_true), jax.random.split(key, M)),
        )
        grads = jax.tree_util.tree_map(lambda g: g / M, grads)
        loss, mean_norm = loss / M, mean_norm / M
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, mean_norm


@eqx.filter_jit
def eval_loss(model, features, weights, x_true, key):
    return gold_loss(model, features, weights, x_true, key)


@eqx.filter_jit
def weighted_mean_rmse(features, weights, x_true, scale=1.0):
    """RMSE of the weighted particle mean against the truth; `scale` maps normalized
    coordinates back to state units."""
    w = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-30)
    mean = jnp.sum(features * w[..., None], axis=-2)
    return jnp.sqrt(jnp.mean((mean - x_true) ** 2)) * scale


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------


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
    normalize_proposals(tcfg)
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

    proposals = list(tcfg.rollout_proposals)
    log_path = os.path.join(cfg.save_path, 'train_log.jsonl')
    rollout_idx = step // tcfg.grad_steps_per_rollout
    t_start = time.time()

    while step < tcfg.num_iterations:
        # Keyed by rollout index so a resumed run replays the same episode stream.
        rk, tk, vk = jax.random.split(jax.random.fold_in(roll_key, rollout_idx), 3)
        rmodel = rollout_model_from(model, cfg.model, tcfg.rollout_fm_steps, tcfg.rollout_bf16, rk)

        t0 = time.time()
        buffer, rstats = collect_rollout(env, rmodel, proposals, tcfg, simulate_episodes, rk)
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

        record = {
            'step': step,
            'rollout': rollout_idx,
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
            record['val'] = validate(model, rmodel, val_eps, proposals, tcfg, vk, rmse_scale)

        with open(log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        every = max(1, tcfg.log_interval // tcfg.grad_steps_per_rollout)
        if rollout_idx % every == 0 or 'val' in record:
            print(json.dumps(record))

    save_checkpoint(cfg.save_path, model, opt_state, step, numbered=False)
    print(f'Model saved to {os.path.join(cfg.save_path, "model.eqx")}')


def validate(model, rollout_model, val_eps: Episodes, proposals, tcfg, key, rmse_scale=1.0):
    """Gold loss and weighted-mean RMSE on the fixed validation episodes, per proposal."""
    x_all = val_eps.x_norm
    B, T = x_all.shape[0], x_all.shape[1] - 1
    x_norm = x_all.reshape(B * (T + 1), -1)
    out = {}
    for i, proposal in enumerate(proposals):
        k = jax.random.fold_in(key, 1000 + i)
        rk, sk, lk = jax.random.split(k, 3)
        feats, ws, neffs = rollout_sets(
            rollout_model,
            proposal,
            tcfg,
            val_eps,
            jax.random.split(rk, B),
            jax.random.split(sk, B * T).reshape(B, T, -1),
        )
        n, D = feats.shape[-2], feats.shape[-1]
        feats = feats.reshape(-1, n, D)
        ws = ws.reshape(-1, n)
        # Chunk the loss to bound memory; each chunk is a jitted call of fixed shape.
        chunk = max(1, tcfg.batch_size // tcfg.micro_batches)
        rows = feats.shape[0]
        losses = []
        for s in range(0, rows - rows % chunk, chunk):
            lk, ck = jax.random.split(lk)
            sl = slice(s, s + chunk)
            losses.append(float(eval_loss(model, feats[sl], ws[sl], x_norm[sl], ck)[0]))
        out[proposal] = {
            'gold_loss': float(np.mean(losses)),
            'wmean_rmse': float(weighted_mean_rmse(feats, ws, x_norm, rmse_scale)),
            'neff_over_n': float(jnp.mean(neffs)) / tcfg.rollout_num_particles,
            'resample_frac': float(jnp.mean(neffs < 0.5 * tcfg.rollout_num_particles)),
        }
    return out
