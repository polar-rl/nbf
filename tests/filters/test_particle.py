"""Tests for ``nbf.filters.particle.ParticleSet``.

Scope is deliberately narrow: ``neff`` and ``systematic_resample`` are pure functions
of weights (plus a key) and don't need an env. The env-coupled methods
(``simulate``, ``ParticleFilter.update`` / ``reset``) are covered indirectly by the
PE integration test and aren't re-tested here.
"""

import jax
import jax.numpy as jnp
import pytest

from nbf.filters.particle import ParticleSet


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def _make(weights):
    """Build a ParticleSet with arbitrary jax-array particles matching the weight count."""
    weights = jnp.asarray(weights)
    particles = jnp.arange(weights.shape[0], dtype=jnp.float32)
    return ParticleSet(particles=particles, weights=weights)


class TestNeff:
    def test_uniform_weights_give_n(self):
        n = 32
        ps = _make(jnp.ones(n) / n)
        assert float(ps.neff()) == pytest.approx(n, rel=1e-5)

    def test_fully_concentrated_gives_1(self):
        weights = jnp.zeros(16).at[3].set(1.0)
        ps = _make(weights)
        assert float(ps.neff()) == pytest.approx(1.0, rel=1e-5)

    def test_all_zero_weights_returns_0(self):
        # The defended degenerate branch — should not produce inf/NaN.
        ps = _make(jnp.zeros(8))
        assert float(ps.neff()) == 0.0


class TestSystematicResample:
    def test_preserves_particle_count(self, key):
        weights = jax.random.dirichlet(key, jnp.ones(20))
        ps = _make(weights)
        resampled = ps.systematic_resample(key)
        assert resampled.weights.shape == (20,)
        assert resampled.particles.shape == (20,)

    def test_output_weights_are_uniform(self, key):
        weights = jax.random.dirichlet(key, jnp.ones(20))
        ps = _make(weights)
        resampled = ps.systematic_resample(key)
        assert jnp.allclose(resampled.weights, 1.0 / 20.0)

    def test_concentrates_on_high_weight_particles(self, key):
        # Weight 90% on particle index 0, 10% spread across the other 9.
        n = 10
        weights = jnp.concatenate([jnp.array([0.9]), jnp.full(n - 1, 0.1 / (n - 1))])
        ps = _make(weights)

        # Average over many keys so the systematic-resample stride doesn't bias us.
        counts_zero = 0
        trials = 64
        for sub in jax.random.split(key, trials):
            resampled = ps.systematic_resample(sub)
            counts_zero += int(jnp.sum(resampled.particles == 0.0))
        # Expected ~9 of 10 picks per trial. Loose bound to avoid flakiness.
        avg = counts_zero / trials
        assert avg > 7.5
