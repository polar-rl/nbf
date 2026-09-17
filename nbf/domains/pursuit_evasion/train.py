"""On-policy training of the pursuit-evasion NBF embedding (see nbf.train for the loop).

Each simulated episode draws k evader policies from the training mixture
(`policies.sample_in_dist_tuple`), so the env differs per episode: the policy leaves
are stacked along a batch axis and the filter rollout is vmapped over them
(`Episodes.env_axes` marks those leaves). The loss is the generator objective at the
true joint position in the model's normalized coordinates.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field

import jax
import jax.numpy as jnp
from embedex.distributions.config import ModelConfig
from jax import Array

from nbf.config import apply_overrides, build_dataclass
from nbf.domains.pursuit_evasion.config import DataConfig
from nbf.domains.pursuit_evasion.env import (
    L,
    PursuitEvasionEnv,
    normalize_positions,
)
from nbf.domains.pursuit_evasion.policies import sample_in_dist_tuple
from nbf.filters.likelihood import likelihood_feature_dim
from nbf.train import Episodes, OnPolicyTrainingConfig, normalize_proposals
from nbf.train import run as _run


@dataclass
class TrainingConfig(OnPolicyTrainingConfig):
    batch_size: int = 256
    rollout_proposals: list = field(default_factory=lambda: ['bootstrap'])


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    save_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    seed: int = 0


def _src_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def load_config(path: str | None, overrides: list[str]) -> Config:
    if path is None:
        cfg = Config()
    else:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = build_dataclass(Config, raw)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    normalize_proposals(cfg.training)
    if not os.path.isabs(cfg.save_path):
        cfg.save_path = os.path.normpath(os.path.join(_src_root(), cfg.save_path))
    return cfg


def build_env(data_cfg: DataConfig) -> PursuitEvasionEnv:
    """The training env without a policy; episodes supply their own (see below)."""
    return PursuitEvasionEnv(
        num_evaders=data_cfg.num_evaders,
        time_limit=data_cfg.episode_length,
        policy=None,
        shuffle_observations=data_cfg.shuffle_observations,
        interaction_radius=data_cfg.interaction_radius,
        interaction_strength=data_cfg.interaction_strength,
    )


def sample_policy_batch(num_episodes: int, num_evaders: int, key: Array) -> tuple:
    """k stacked Policy modules, each with a leading axis of `num_episodes` iid draws."""
    tuples = [sample_in_dist_tuple(k, num_evaders) for k in jax.random.split(key, num_episodes)]
    return tuple(
        jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *[t[j] for t in tuples])
        for j in range(num_evaders)
    )


def env_with_policies(env: PursuitEvasionEnv, policies_b: tuple) -> tuple:
    """Attach batched policies to `env`; returns (env_b, env_axes) for a vmap over episodes."""
    import equinox as eqx

    env_b = eqx.tree_at(lambda e: e.policy, env, policies_b)

    def _axis(path, _):
        return 0 if any(getattr(p, 'name', None) == 'policy' for p in path) else None

    env_axes = jax.tree_util.tree_map_with_path(_axis, env_b)
    return env_b, env_axes


def _rollout_one(env: PursuitEvasionEnv, key: Array):
    reset_key, scan_key = jax.random.split(key)
    state0, init_ts = env.reset(reset_key)
    step_keys = jax.random.split(scan_key, env.time_limit)

    def _step(state, key_t):
        next_state, timestep = env.step(state, 0, key_t)
        return next_state, (next_state, timestep.observation)

    _, (states_tail, obs_tail) = jax.lax.scan(_step, state0, step_keys)
    prepend = lambda head, tail: jnp.concatenate([jnp.asarray(head)[None], tail], axis=0)  # noqa: E731
    return jax.tree.map(prepend, state0, states_tail), jax.tree.map(prepend, init_ts.observation, obs_tail)


def simulate_episodes(env: PursuitEvasionEnv, num_episodes: int, key: Array) -> Episodes:
    """Episodes under freshly drawn training-mixture policies; x_norm is (B, T + 1, 2k)."""
    pol_key, ep_key = jax.random.split(key)
    policies_b = sample_policy_batch(num_episodes, env.num_evaders, pol_key)
    env_b, env_axes = env_with_policies(env, policies_b)
    states_all, obs_all = jax.vmap(_rollout_one, in_axes=(env_axes, 0))(
        env_b, jax.random.split(ep_key, num_episodes)
    )
    B, T1 = states_all.positions.shape[:2]
    return Episodes(
        state0=jax.tree.map(lambda x: x[:, 0], states_all),
        obs0=jax.tree.map(lambda x: x[:, 0], obs_all),
        states_next=jax.tree.map(lambda x: x[:, 1:], states_all),
        obs_prev=jax.tree.map(lambda x: x[:, :-1], obs_all),
        obs_next=jax.tree.map(lambda x: x[:, 1:], obs_all),
        x_norm=normalize_positions(states_all.positions).reshape(B, T1, -1),
        env=env_b,
        env_axes=env_axes,
    )


def run(cfg: Config) -> None:
    env = build_env(cfg.data)
    cfg.model.output_dim = 2 * int(cfg.data.num_evaders)
    cfg.model.input_dim = cfg.model.output_dim + likelihood_feature_dim(env)
    _run(cfg, env=env, simulate_episodes=simulate_episodes, rmse_scale=0.5 * L)


def main() -> None:
    parser = argparse.ArgumentParser(description='On-policy training of the pursuit-evasion NBF.')
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
