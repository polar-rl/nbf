import argparse
import os
from dataclasses import dataclass, field

import yaml
from embedex.distributions.config import ModelConfig

from nbf.config import TrainingConfig as _BaseTrainingConfig
from nbf.config import load_config as _load_config


def _src_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def resolve_runtime_paths(cfg: 'Config') -> 'Config':
    base = _src_root()

    if not os.path.isabs(cfg.save_path):
        cfg.save_path = os.path.normpath(os.path.join(base, cfg.save_path))
    if not os.path.isabs(cfg.data_path):
        cfg.data_path = os.path.normpath(os.path.join(base, cfg.data_path))
    if cfg.load_model is not None and not os.path.isabs(cfg.load_model):
        cfg.load_model = os.path.normpath(os.path.join(base, cfg.load_model))

    return cfg


@dataclass
class DataConfig:
    K: int = 20
    num_scenarios: int = 5000
    block_size: int = 50
    episode_length: int = 20
    num_pf_particles: int = 100000
    dt: float = 0.01
    window_substeps: int = 5
    sigma_dyn: float = 0.25
    sigma_obs_min: float = 0.5
    sigma_obs_max: float = 1.5
    f_bands: list = field(default_factory=lambda: [[5.0, 7.0], [9.0, 11.0]])
    spinup_time: float = 10.0
    sigma_init: float = 2.0
    num_val_scenarios: int = 250
    norm_mean: float | None = None
    norm_std: float | None = None


@dataclass
class TrainingConfig(_BaseTrainingConfig):
    batch_size: int = 256
    num_samples: int = 2048


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    save_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    load_model: str | None = None
    seed: int = 0


def load_config(path: str) -> Config:
    cfg = _load_config(Config, path)

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    for key in ('save_path', 'data_path'):
        if key in raw and not os.path.isabs(raw[key]):
            setattr(cfg, key, raw[key])
    if cfg.load_model is not None and not os.path.isabs(cfg.load_model) and 'load_model' in raw:
        cfg.load_model = raw['load_model']
    return resolve_runtime_paths(cfg)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description='Train Lorenz-96 NBF embedding model.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument(
        '--set', nargs='*', default=[], help='Override config values: key.path=value'
    )
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config is not None else Config()

    if args.set:
        from nbf.config import apply_overrides

        cfg = apply_overrides(cfg, args.set)

    return resolve_runtime_paths(cfg)
