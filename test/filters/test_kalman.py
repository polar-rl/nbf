"""Tests for ``nbf.filters.kalman.UKF``.

UKF is now domain-agnostic — we test it with a synthetic identity dynamics and a
range-only observation from a fixed sensor. No PE coupling.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from nbf.filters.kalman import UKF


class _Obs(eqx.Module):
    """Minimal observation stand-in matching the ``.ranges`` interface UKF.update reads."""

    ranges: jax.Array


def _identity_propagate(position, policy, obstacles):
    """Trivial dynamics: state doesn't move."""
    del policy, obstacles
    return position


@pytest.fixture
def ukf():
    return UKF(
        num_evaders=1,
        sigma_dyn=0.1,
        sigma_obs=0.05,
        sensor_pos=jnp.array([0.0, 0.0]),
        propagate_fn=_identity_propagate,
    )


class TestUpdate:
    def test_observation_shrinks_uncertainty(self, ukf):
        # Wide prior: high uncertainty over a small region.
        mu_pred = jnp.array([3.0, 0.0])
        sigma_pred = jnp.eye(2) * 4.0
        # Observe the range from the sensor at origin to the true position (3, 0): exactly 3.
        obs = _Obs(ranges=jnp.array([3.0]))

        _, sigma_post = ukf.update(mu_pred, sigma_pred, obs)

        # Posterior trace should be smaller than prior trace.
        assert float(jnp.trace(sigma_post)) < float(jnp.trace(sigma_pred))

    def test_observation_pulls_mean_consistent_with_range(self, ukf):
        # Prior says we are at (2, 0); observation implies range 4 → true position is further out.
        mu_pred = jnp.array([2.0, 0.0])
        sigma_pred = jnp.eye(2) * 1.0
        obs = _Obs(ranges=jnp.array([4.0]))

        mu_post, _ = ukf.update(mu_pred, sigma_pred, obs)

        # Post-update range should be closer to 4 than the prior range of 2.
        post_range = float(jnp.linalg.norm(mu_post))
        assert post_range > 2.0
