"""Lorenz-96 domain settings. Training and run configs live in lorenz96/train.py."""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    K: int = 20
    episode_length: int = 20
    dt: float = 0.01
    window_substeps: int = 5
    sigma_dyn: float = 0.25
    sigma_obs_min: float = 0.5
    sigma_obs_max: float = 1.5
    f_bands: list = field(default_factory=lambda: [[5.0, 7.0], [9.0, 11.0]])
    spinup_time: float = 10.0
    sigma_init: float = 2.0
    norm_mean: float | None = None
    norm_std: float | None = None
