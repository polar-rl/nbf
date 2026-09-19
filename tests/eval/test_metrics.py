import jax.numpy as jnp
import numpy as np
import pytest

from nbf.eval.base import chunk_bounds, chunked_stream, parse_method_spec, run_chunked
from nbf.eval.metrics import (
    coverage_per_step_batch,
    crps_per_step_batch,
    rmse_per_step_batch,
    sample_metrics_per_step,
    spread_per_step_batch,
)


class TestParseMethodSpec:
    @pytest.mark.parametrize(
        ('spec', 'expected'),
        [
            ('nbf_256', ('nbf', 256)),
            ('pf_8192', ('pf', 8192)),
            ('gmf_4', ('gmf', 4)),
            ('pf_reg_1024', ('pf_reg', 1024)),
            ('pf_rm_1024', ('pf_rm', 1024)),
            ('enkf_20', ('enkf', 20)),
        ],
    )
    def test_specs(self, spec, expected):
        assert parse_method_spec(spec) == expected

    @pytest.mark.parametrize('spec', ['1024', 'pf_', 'pf_x'])
    def test_invalid(self, spec):
        with pytest.raises(ValueError):
            parse_method_spec(spec)


class TestRmsePerStepBatch:
    def test_exact_mean_gives_zero(self):
        truth = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
        samples = jnp.broadcast_to(truth[:, :, None, :], (2, 3, 8, 4))
        rmse = np.asarray(rmse_per_step_batch(samples, truth))
        assert rmse.shape == (2, 3)
        assert np.allclose(rmse, 0.0)

    def test_constant_offset(self):
        truth = jnp.zeros((1, 2, 5))
        samples = jnp.full((1, 2, 16, 5), 2.0)
        rmse = np.asarray(rmse_per_step_batch(samples, truth))
        assert np.allclose(rmse, 2.0)

    def test_symmetric_samples_cancel(self):
        truth = jnp.zeros((1, 1, 3))
        samples = jnp.stack([jnp.ones((1, 1, 3)), -jnp.ones((1, 1, 3))], axis=2)
        rmse = np.asarray(rmse_per_step_batch(samples, truth))
        assert np.allclose(rmse, 0.0)


def _gaussian_crps(mu, sigma, x):
    # Closed form for N(mu, sigma^2): sigma [z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi)].
    from scipy.stats import norm

    z = (x - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


class TestCrpsPerStepBatch:
    def test_point_mass_is_absolute_error(self):
        truth = jnp.zeros((1, 1, 3))
        samples = jnp.full((1, 1, 16, 3), 2.0)
        crps = np.asarray(crps_per_step_batch(samples, truth))
        assert crps.shape == (1, 1)
        assert np.allclose(crps, 2.0)

    def test_matches_brute_force_pairs(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(2, 3, 7, 4)).astype(np.float32)
        x = rng.normal(size=(2, 3, 4)).astype(np.float32)
        crps = np.asarray(crps_per_step_batch(jnp.asarray(X), jnp.asarray(x)))
        S = X.shape[2]
        term1 = np.abs(X - x[:, :, None, :]).mean(axis=2)
        diffs = np.abs(X[:, :, :, None, :] - X[:, :, None, :, :]).sum(axis=(2, 3))
        term2 = 0.5 * diffs / (S * (S - 1))  # fair estimator: distinct pairs only
        expected = (term1 - term2).mean(axis=-1)
        assert np.allclose(crps, expected, atol=1e-5)

    def test_gaussian_closed_form(self):
        pytest.importorskip('scipy')
        mu, sigma = 0.3, 1.7
        truths = np.array([-1.0, 0.3, 2.5])
        rng = np.random.default_rng(1)
        S = 20000
        X = rng.normal(mu, sigma, size=(1, 3, S, 1)).astype(np.float32)
        crps = np.asarray(crps_per_step_batch(jnp.asarray(X), jnp.asarray(truths.reshape(1, 3, 1))))
        expected = _gaussian_crps(mu, sigma, truths)
        assert np.allclose(crps[0], expected, atol=0.03)


class TestCoverageAndSpread:
    def test_coverage_of_calibrated_gaussian(self):
        rng = np.random.default_rng(2)
        N, T, S, D = 1, 1, 4000, 2000
        X = rng.normal(size=(N, T, S, D)).astype(np.float32)
        x = rng.normal(size=(N, T, D)).astype(np.float32)
        for level in (0.5, 0.9):
            cov = np.asarray(coverage_per_step_batch(jnp.asarray(X), jnp.asarray(x), level))
            assert cov.shape == (N, T)
            assert abs(float(cov[0, 0]) - level) < 0.04

    def test_coverage_zero_when_truth_outside(self):
        X = jnp.zeros((1, 1, 8, 3)) + jnp.arange(8.0)[None, None, :, None]
        x = jnp.full((1, 1, 3), 100.0)
        assert np.allclose(coverage_per_step_batch(X, x, 0.9), 0.0)

    def test_spread_is_sample_std(self):
        rng = np.random.default_rng(3)
        X = rng.normal(0, 2.0, size=(1, 1, 5000, 3)).astype(np.float32)
        spread = np.asarray(spread_per_step_batch(jnp.asarray(X)))
        assert abs(float(spread[0, 0]) - 2.0) < 0.1

    def test_sample_metrics_keys_and_shapes(self):
        X = jnp.asarray(np.random.default_rng(4).normal(size=(2, 3, 9, 4)).astype(np.float32))
        x = jnp.zeros((2, 3, 4))
        out = sample_metrics_per_step(X, x)
        assert set(out) == {'rmse', 'crps', 'spread', 'cov50', 'cov90'}
        assert all(v.shape == (2, 3) for v in out.values())


class TestChunkBounds:
    def test_single_chunk_when_size_covers_all(self):
        assert chunk_bounds(10, 0) == [(0, 10)]
        assert chunk_bounds(10, 10) == [(0, 10)]
        assert chunk_bounds(10, 99) == [(0, 10)]

    def test_ragged_tail(self):
        assert chunk_bounds(7, 3) == [(0, 3), (3, 6), (6, 7)]


class TestRunChunked:
    def test_matches_unchunked(self):
        x = jnp.arange(12.0).reshape(6, 2)
        fn = lambda s, e: x[s:e] * 2.0  # noqa: E731
        full = np.asarray(run_chunked(fn, 6, 0))
        for cs in (1, 4, 6):
            assert np.array_equal(np.asarray(run_chunked(fn, 6, cs)), full)

    def test_handles_tuple_returns(self):
        x = jnp.arange(6.0)
        fn = lambda s, e: (x[s:e], x[s:e] ** 2)  # noqa: E731
        a, b = run_chunked(fn, 6, 2)
        assert np.array_equal(np.asarray(a), np.arange(6.0))
        assert np.array_equal(np.asarray(b), np.arange(6.0) ** 2)

    def test_chunked_stream_slices_trees_and_keys(self):
        tree = {'a': jnp.arange(8.0), 'b': jnp.arange(8.0) * 10}
        keys = jnp.arange(8.0) + 100
        stream = lambda t, k, scale: (t['a'] + t['b'] + k) * scale  # noqa: E731
        full = np.asarray(chunked_stream(stream, (tree,), (keys,), (2.0,), 8, 0))
        got = np.asarray(chunked_stream(stream, (tree,), (keys,), (2.0,), 8, 3))
        assert np.array_equal(got, full)


class TestRelabelToTruth:
    def _perms(self, k):
        import itertools

        return jnp.asarray(np.array(list(itertools.permutations(range(k))), dtype=np.int32))

    def test_swapped_labels_recover_zero_rmse(self):
        from nbf.eval.metrics import relabel_samples_to_truth

        truth = jnp.asarray(np.random.default_rng(0).normal(size=(2, 3, 2, 2)))  # N,T,k,d
        swapped = truth[:, :, None, ::-1, :]
        swapped = jnp.broadcast_to(swapped, (2, 3, 4, 2, 2))
        out = relabel_samples_to_truth(swapped, truth, self._perms(2))
        assert out.shape == swapped.shape
        assert np.allclose(np.asarray(out), np.asarray(truth[:, :, None]))
        assert float(rmse_per_step_batch(out, truth).max()) < 1e-6
        assert float(rmse_per_step_batch(swapped, truth).min()) > 0.1

    def test_relabelling_never_hurts_and_keeps_multiset(self):
        from nbf.eval.metrics import relabel_samples_to_truth

        rng = np.random.default_rng(1)
        X = jnp.asarray(rng.normal(size=(3, 4, 16, 3, 2)))
        x = jnp.asarray(rng.normal(size=(3, 4, 3, 2)))
        out = relabel_samples_to_truth(X, x, self._perms(3))
        d_before = jnp.sum((X - x[:, :, None]) ** 2, axis=(-1, -2))
        d_after = jnp.sum((out - x[:, :, None]) ** 2, axis=(-1, -2))
        assert bool(jnp.all(d_after <= d_before + 1e-6))
        # every relabelled sample is a permutation of the original sample
        assert np.allclose(np.sort(np.asarray(out), axis=3), np.sort(np.asarray(X), axis=3))

    def test_sample_metrics_accept_perms(self):
        rng = np.random.default_rng(2)
        X = jnp.asarray(rng.normal(size=(2, 3, 8, 2, 2)))
        x = jnp.asarray(rng.normal(size=(2, 3, 2, 2)))
        plain = sample_metrics_per_step(X, x)
        matched = sample_metrics_per_step(X, x, relabel_perms=self._perms(2))
        assert set(plain) == set(matched)
        assert all(matched[name].shape == plain[name].shape for name in plain)
        assert not np.allclose(matched['rmse'], plain['rmse'])
        single = sample_metrics_per_step(X, x, relabel_perms=self._perms(1)[:, :1])
        assert np.allclose(single['rmse'], plain['rmse'])
