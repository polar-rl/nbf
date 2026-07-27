import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nbf.eval.base import chunk_bounds, chunked_stream, parse_method_spec, run_chunked
from nbf.eval.metrics import (
    mmd2_multibw_per_step_batch,
    mmd2_multibw_per_step_chunked,
    rmse_per_step_batch,
)


class TestParseMethodSpec:
    @pytest.mark.parametrize(
        ('spec', 'expected'),
        [
            ('pf_oracle', ('pf_oracle', 0)),
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


class TestMmdChunking:
    def test_chunked_matches_batched(self):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        method = jax.random.normal(k1, (5, 3, 64, 4))
        oracle = jax.random.normal(k2, (5, 3, 64, 4)) + 0.3
        full = np.asarray(mmd2_multibw_per_step_batch(method, oracle))
        for cs in (1, 2, 5, 0):
            got = np.asarray(mmd2_multibw_per_step_chunked(method, oracle, chunk_size=cs))
            assert got.shape == (5, 3)
            assert np.allclose(got, full, rtol=1e-3, atol=1e-5), cs
