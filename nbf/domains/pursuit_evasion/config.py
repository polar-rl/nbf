"""Pursuit-evasion domain settings. Training and run configs live in pursuit_evasion/train.py."""

from dataclasses import dataclass


@dataclass
class DataConfig:
    num_evaders: int = 1
    episode_length: int = 20
    shuffle_observations: bool = False
    num_sensors: int = 1
    interaction_radius: float = 1.5
    interaction_strength: float = 1.0
