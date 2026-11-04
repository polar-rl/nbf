import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from embedex.config import build_dataclass
from embedex.distributions.builder import create_model
from embedex.distributions.config import ModelConfig
from jax import Array
from tqdm import tqdm


def batch_size(tree) -> int:
    return jax.tree.leaves(tree)[0].shape[0]


def time_len(tree) -> int:
    return jax.tree.leaves(tree)[0].shape[1]


@eqx.filter_jit
def env_rollout_batch(env, episode_length, keys):
    def _one(key):
        reset_key, scan_key = jax.random.split(key)
        state0, init_ts = env.reset(reset_key)
        obs0 = init_ts.observation
        step_keys = jax.random.split(scan_key, episode_length)

        def _step(state, key_t):
            next_state, timestep = env.step(state, 0, key_t)
            return next_state, (next_state, timestep.observation)

        _, (states_tail, obs_tail) = jax.lax.scan(_step, state0, step_keys)

        def _prepend(head, tail):
            head_arr = jnp.asarray(head)
            return jnp.concatenate([head_arr[None], tail], axis=0)

        states_all = jax.tree.map(_prepend, state0, states_tail)
        obs_all = jax.tree.map(_prepend, obs0, obs_tail)
        return states_all, obs_all

    return jax.vmap(_one)(keys)


def slice_rollout_no_gt(states_all, obs_all):
    obs_prev = jax.tree.map(lambda x: x[:, :-1], obs_all)
    obs_next = jax.tree.map(lambda x: x[:, 1:], obs_all)
    states_next = jax.tree.map(lambda x: x[:, 1:], states_all)
    state0 = jax.tree.map(lambda x: x[:, 0], states_all)
    obs0 = jax.tree.map(lambda x: x[:, 0], obs_all)
    return obs_prev, obs_next, states_next, state0, obs0


def parse_seeds(args) -> list[int]:
    raw = (getattr(args, 'seeds', '') or '').strip()
    if raw:
        return [int(s) for s in raw.split(',') if s.strip()]
    n = int(getattr(args, 'num_seeds', 0) or 0)
    base = int(getattr(args, 'seed', 0) or 0)
    if n > 1:
        return [base + i for i in range(n)]
    return [base]


def load_embedding_model(model_dir, key, fm_num_steps: int | None = None):
    config_path = os.path.join(model_dir, 'model_config.json')
    model_path = os.path.join(model_dir, 'model.eqx')

    with open(config_path) as f:
        mcfg = json.load(f)
    if fm_num_steps:
        mcfg['generator']['fm_num_steps'] = int(fm_num_steps)

    model_cfg = build_dataclass(ModelConfig, mcfg)
    model = create_model(model_cfg, dequantize=False, key=key)
    model = jax.tree.map(
        lambda x: (
            x.astype(jnp.float32)
            if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)
            else x
        ),
        model,
    )
    with open(model_path, 'rb') as f:
        model = eqx.tree_deserialise_leaves(f, model)
    return model, model_cfg


def cast_velocity_net_bf16(model):
    vnet = model.generator.velocity_net
    vnet16 = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, vnet)
    return eqx.tree_at(lambda m: m.generator.velocity_net, model, vnet16)


def parse_method_spec(spec: str) -> tuple[str, int]:
    kind, _, n = spec.rpartition('_')
    if not kind:
        raise ValueError(f'Invalid method spec {spec!r}')
    return kind, int(n)


def rollout_episodes(env, episode_length: int, num_episodes: int, key: Array):
    ep_keys = jax.random.split(key, num_episodes)
    states_all, obs_all = env_rollout_batch(env, episode_length, ep_keys)
    obs_prev_b, obs_next_b, states_next_b, state0_b, obs0_b = slice_rollout_no_gt(
        states_all, obs_all
    )
    return state0_b, obs0_b, states_next_b, obs_prev_b, obs_next_b


def split_T(key: Array, N: int, T: int):
    return jax.random.split(key, N * T).reshape(N, T, -1)


def tree_slice(tree, start: int, stop: int):
    """Slice every leaf of ``tree`` along the leading (episode) axis."""
    return jax.tree.map(lambda x: x[start:stop], tree)


def chunk_bounds(n: int, chunk_size: int) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` covering ``range(n)``; a non-positive size means one chunk."""
    if chunk_size <= 0 or chunk_size >= n:
        return [(0, n)]
    return [(s, min(s + chunk_size, n)) for s in range(0, n, chunk_size)]


def run_chunked(fn, n: int, chunk_size: int, *, desc: str | None = None):
    """Run ``fn(start, stop)`` over leading-axis chunks and concatenate the results.

    Each chunk is pulled to host memory as it completes, so peak device memory
    scales with ``chunk_size`` rather than ``n``. A single chunk is passed straight
    through untouched (device arrays, no host copy). ``fn`` may return any pytree of arrays.
    """
    bounds = chunk_bounds(n, chunk_size)
    if len(bounds) == 1:
        return fn(0, n)
    outs = []
    for start, stop in tqdm(bounds, desc=desc, unit='chunk', disable=desc is None):
        outs.append(jax.tree.map(np.asarray, fn(start, stop)))
    return jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *outs)


def chunked_stream(
    stream_fn,
    episode_trees,
    key_arrays,
    tail_args,
    n: int,
    chunk_size: int,
    *,
    desc: str | None = None,
):
    """Chunk a ``*_stream_batch`` call over its leading episode axis.

    ``stream_fn`` is invoked as ``stream_fn(*episode_trees, *key_arrays, *tail_args)``
    with the episode trees and key arrays sliced to the current chunk. Keys are split
    for the full batch by the caller and only sliced here, so per-episode randomness --
    and therefore the result -- does not depend on ``chunk_size``.
    """

    def _run(start: int, stop: int):
        trees = [tree_slice(t, start, stop) for t in episode_trees]
        keys = [k[start:stop] for k in key_arrays]
        return stream_fn(*trees, *keys, *tail_args)

    return run_chunked(_run, n, chunk_size, desc=desc)


def parse_csv_ks(args, attr: str, default: tuple[int, ...]) -> tuple[int, ...]:
    val = getattr(args, attr, '')
    if val:
        return tuple(int(x.strip()) for x in val.split(',') if x.strip())
    return default


def parse_csv_methods(args, attr: str, default: tuple[str, ...]) -> tuple[str, ...]:
    val = getattr(args, attr, '')
    if val:
        return tuple(m.strip() for m in val.split(',') if m.strip())
    return default


def load_model_for_k(args, k: int):
    mapping = getattr(args, 'model_dirs_per_k_dict', None) or {}
    md = mapping.get(k)
    if md is None:
        return None
    em, _ = load_embedding_model(md, jax.random.PRNGKey(0))
    return em
