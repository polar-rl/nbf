"""On-policy training of the Lorenz-96 NBF embedding (see nbf.train for the loop).

Episodes are simulated on the fly and the filter is rolled out along them with the
current parameters.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field

import equinox as eqx
import jax
import numpy as np
from embedex.distributions.config import ModelConfig
from jax import Array

from nbf.config import apply_overrides, build_dataclass
from nbf.domains.lorenz96.config import DataConfig
from nbf.domains.lorenz96.env import Lorenz96Env, build_lorenz96_env
from nbf.eval.base import env_rollout_batch
from nbf.train import Episodes, OnPolicyTrainingConfig
from nbf.train import run as _run


@dataclass
class TrainingConfig(OnPolicyTrainingConfig):
    batch_size: int = 256
    norm_episodes: int = 512  # episodes used to estimate the climatological norm stats


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
    if not os.path.isabs(cfg.save_path):
        cfg.save_path = os.path.normpath(os.path.join(_src_root(), cfg.save_path))
    return cfg


def estimate_norm_stats(env: Lorenz96Env, num_episodes: int, key: Array) -> tuple[float, float]:
    """Climatological mean/std of the true state over simulated episodes."""
    states_all, _ = env_rollout_batch(env, env.time_limit, jax.random.split(key, num_episodes))
    x = np.asarray(states_all.x, dtype=np.float64)
    return float(x.mean()), float(x.std())


@eqx.filter_jit
def simulate_episodes(env: Lorenz96Env, num_episodes: int, key: Array) -> Episodes:
    """One env serves every episode (env_axes None); x_norm is (B, T + 1, K)."""
    states_all, obs_all = env_rollout_batch(
        env, env.time_limit, jax.random.split(key, num_episodes)
    )
    return Episodes(
        state0=jax.tree.map(lambda x: x[:, 0], states_all),
        obs0=jax.tree.map(lambda x: x[:, 0], obs_all),
        states_next=jax.tree.map(lambda x: x[:, 1:], states_all),
        obs_prev=jax.tree.map(lambda x: x[:, :-1], obs_all),
        obs_next=jax.tree.map(lambda x: x[:, 1:], obs_all),
        x_norm=(states_all.x - env.norm_mean) / env.norm_std,
        env=env,
        env_axes=None,
    )


def run(cfg: Config) -> None:
    tcfg = cfg.training
    key = jax.random.PRNGKey(cfg.seed)
    norm_key = jax.random.fold_in(key, 7)
    # Norm stats ride in data_config.json so run_grid --model_dir works unchanged.
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
    cfg.model.input_dim = int(cfg.data.K)
    cfg.model.likelihood_dim = env.likelihood_feature_dim
    _run(cfg, env=env, simulate_episodes=simulate_episodes, rmse_scale=cfg.data.norm_std)


def main() -> None:
    parser = argparse.ArgumentParser(description='On-policy training of the Lorenz-96 NBF.')
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
