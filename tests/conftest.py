"""Fixtures shared by every test.

Every domain is registered once in `DOMAINS`. Tests that take the `domain` fixture run for each
of them, so a new domain only needs an entry here; structure specific to one domain is tested
next to it in `tests/domains/<domain>/`.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import pytest

from nbf.domains.lorenz96 import train as lorenz96
from nbf.domains.lorenz96.env import build_lorenz96_env
from nbf.domains.pursuit_evasion import train as pursuit_evasion
from nbf.domains.pursuit_evasion.policies import make_random_walk_policy

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'nbf', 'domains', '{}', 'configs')


def random_walk_evaders(env):
    """The pursuit-evasion training env with every evader on a random walk. The trainer hands
    each episode its own policies, so its env has none and cannot run an episode alone."""

    return eqx.tree_at(lambda e: e.policy, env, (make_random_walk_policy(),) * env.num_evaders)


@dataclass(frozen=True)
class Domain:
    name: str
    module: Any  # the domain's `train` module: load_config, simulate_episodes, run
    build_env: Callable[[Any], Any]  # data config -> the env the trainer uses
    smoke_config: str  # a few iterations of training on a small instance, under configs/
    standalone: Callable[[Any], Any] = lambda env: env  # trainer env -> one that runs alone

    def config(self, overrides: list[str] = ()) -> Any:
        """The domain's smoke config with `key.path=value` overrides applied."""

        path = os.path.join(CONFIG_DIR.format(self.name), self.smoke_config)
        return self.module.load_config(path, list(overrides))

    def env(self, overrides: list[str] = ()) -> Any:
        """An env of the smoke config's size that runs episodes on its own."""

        return self.standalone(self.build_env(self.config(overrides).data))


DOMAINS = (
    Domain(
        name='lorenz96',
        module=lorenz96,
        build_env=build_lorenz96_env,
        smoke_config='smoke_l96.yaml',
    ),
    Domain(
        name='pursuit_evasion',
        module=pursuit_evasion,
        build_env=pursuit_evasion.build_env,
        smoke_config='smoke_k1.yaml',
        standalone=random_walk_evaders,
    ),
)


@pytest.fixture
def key() -> jax.Array:
    # nbf passes raw uint32 keys around (and reshapes them), so tests use them too.
    return jax.random.PRNGKey(0)


@pytest.fixture(scope='session')
def domains() -> dict[str, Domain]:
    """Every domain by name, for tests of one domain (and module-scoped setups)."""

    return {d.name: d for d in DOMAINS}


@pytest.fixture(params=DOMAINS, ids=lambda d: d.name)
def domain(request) -> Domain:
    return request.param
