from abc import abstractmethod
from typing import TypeVar

import equinox as eqx
import jax

S = TypeVar('S', bound='State')


class Observation(eqx.Module):
    pass


class State(eqx.Module):
    @property
    @abstractmethod
    def max_feature_value(self) -> int:
        pass

    @property
    def nfeatures(self) -> int:
        return self.to_flat_features().shape[-1]

    @abstractmethod
    def to_flat_features(self) -> jax.Array:
        pass

    @abstractmethod
    def from_flat_features(self, features: jax.Array) -> S:
        pass

    @abstractmethod
    def observation_from_state(self) -> Observation:
        pass
