import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from embedex.distributions.builder import create_model
from embedex.distributions.config import ModelConfig

from nbf.config import build_dataclass
from nbf.filters.gmf import GMF

from nbf.domains.pursuit_evasion.env import (
    SENSOR_POS,
    SIGMA_DYN,
    SIGMA_OBS_MAX,
    L,
    PursuitEvasionEnv,
    propagate_deterministic,
)


def make_pe_gmf(
    num_evaders: int,
    M: int = 1,
    observation_sigma_scale: float = 1.0,
    kl_merge: float = 0.5,
    weight_prune: float = 0.01,
) -> GMF:
    n = 2 * num_evaders
    return GMF(
        num_evaders=num_evaders,
        sigma_dyn=SIGMA_DYN,
        sigma_obs=SIGMA_OBS_MAX,
        sensor_pos=SENSOR_POS,
        propagate_fn=propagate_deterministic,
        init_mu=jnp.full(n, L / 2),
        init_sigma=((L / 2) ** 2 * jnp.eye(n)),
        M=M,
        jitter_scale=0.1 * L,
        kl_merge=kl_merge,
        weight_prune=weight_prune,
        observation_sigma_scale=observation_sigma_scale,
    )


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


def load_embedding_model(model_dir, key):
    config_path = os.path.join(model_dir, 'model_config.json')
    model_path = os.path.join(model_dir, 'model.eqx')

    with open(config_path) as f:
        mcfg = json.load(f)

    model_cfg = build_dataclass(ModelConfig, mcfg)
    model = create_model(model_cfg, dequantize=False, key=key)
    model = jax.tree.map(
        lambda x: x.astype(jnp.float32)
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)
        else x,
        model,
    )
    with open(model_path, 'rb') as f:
        model = eqx.tree_deserialise_leaves(f, model)
    return model, model_cfg


def make_env_kn(k: int, episode_length: int, policy_tuple) -> PursuitEvasionEnv:
    return PursuitEvasionEnv(
        num_evaders=k,
        time_limit=episode_length,
        policy=policy_tuple,
        shuffle_observations=True,
    )


@eqx.filter_jit
def sample_background_batch(keys_b: Array, M: int, k: int) -> Array:
    return jax.vmap(jax.vmap(lambda kk: jax.random.uniform(kk, (M, k, 2), minval=0.0, maxval=L)))(
        keys_b
    )


def parse_method_spec(spec: str) -> tuple[str, int]:
    if spec == 'pf_oracle':
        return 'pf_oracle', 0
    if spec.startswith('nbf_'):
        return 'nbf', int(spec[len('nbf_') :])
    if spec.startswith('pf_'):
        return 'pf', int(spec[len('pf_') :])
    kind, n = spec.split('_', 1)
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
