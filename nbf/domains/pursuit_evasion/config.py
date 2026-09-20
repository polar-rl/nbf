"""Pursuit-evasion domain settings. Training and run configs live in pursuit_evasion/train.py."""

from dataclasses import dataclass


@dataclass
class DataConfig:
    num_evaders: int = 1
    episode_length: int = 20
    shuffle_observations: bool = False
    num_sensors: int = 1
    obs_interval: int = 1
    sigma_obs_min: float = 0.1  # per-episode range noise drawn uniformly from this interval
    sigma_obs_max: float = 0.5
    interaction_radius: float = 1.5
    interaction_strength: float = 1.0
