"""Tests for ``nbf.filters.gmf.GMF``.

GMF is now domain-agnostic — we construct with synthetic dynamics + sensor.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from nbf.filters.gmf import GMF


class _Obs(eqx.Module):
    ranges: jax.Array


def _identity_propagate(position, policy, obstacles):
    del policy, obstacles
    return position


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def _make_gmf(M: int):
    n = 2  # num_evaders=1 → state dim 2
    return GMF(
        num_evaders=1,
        sigma_dyn=0.1,
        sigma_obs=0.05,
        sensor_pos=jnp.array([0.0, 0.0]),
        propagate_fn=_identity_propagate,
        init_mu=jnp.array([1.0, 1.0]),
        init_sigma=jnp.eye(n) * 1.0,
        M=M,
        jitter_scale=0.5,
    )


class TestInit:
    def test_single_component_has_unit_weight(self, key):
        gmf = _make_gmf(M=1)
        w, mu, sg = gmf.init(key=key)
        assert float(jnp.sum(w)) == pytest.approx(1.0, rel=1e-5)
        assert float(w[0]) == pytest.approx(1.0, rel=1e-5)

    def test_multi_component_weights_sum_to_one_and_are_uniform(self, key):
        gmf = _make_gmf(M=4)
        w, _, _ = gmf.init(key=key)
        assert float(jnp.sum(w)) == pytest.approx(1.0, rel=1e-5)
        assert jnp.allclose(w, 0.25)

    def test_jitter_separates_components(self, key):
        gmf = _make_gmf(M=4)
        _, mus, _ = gmf.init(key=key)
        # With jitter_scale > 0, components should not all be identical.
        max_pairwise = float(jnp.max(jnp.abs(mus - mus[0])))
        assert max_pairwise > 0.0


class TestStep:
    def test_end_to_end_produces_valid_posterior(self, key):
        gmf = _make_gmf(M=2)
        w0, mu0, sg0 = gmf.init(key=key)
        obs = _Obs(ranges=jnp.array([jnp.sqrt(2.0)]))  # distance from origin to (1, 1)

        w, mu, sg = gmf.step(w0, mu0, sg0, policy=None, obstacles=None, observation=obs)

        # Weights stay a probability vector.
        assert float(jnp.sum(w)) == pytest.approx(1.0, abs=1e-3)
        assert bool(jnp.all(w >= 0.0))
        # Mu and sigma stay finite.
        assert bool(jnp.all(jnp.isfinite(mu)))
        assert bool(jnp.all(jnp.isfinite(sg)))
