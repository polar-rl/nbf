"""What every domain's environment offers the filters: episodes that start without an
observation and end at the time limit."""

import jax
import numpy as np
import pytest
from jumanji.types import StepType


@pytest.fixture
def env(domain):
    return domain.env()


def test_an_episode_starts_without_an_observation(env, key):
    _, ts = env.reset(key)
    particles = jax.vmap(env.generator)(jax.random.split(key, 16))

    weights, _ = env.observation_weights(particles, None, ts.observation)

    assert ts.step_type == StepType.FIRST
    assert not bool(env.observation_present(ts.observation))
    np.testing.assert_allclose(weights, 1 / 16, rtol=1e-6)


def test_an_episode_ends_at_the_time_limit(env, key):
    state, ts = env.reset(key)

    for step_key in jax.random.split(key, env.time_limit):
        assert not bool(state.terminal)
        state, ts = env.step(state, 0, step_key)

    assert int(state.step_count) == env.time_limit
    assert bool(state.terminal)
    assert ts.step_type == StepType.LAST
