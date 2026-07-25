import jax.numpy as jnp
import numpy as np
import pytest

from nbf.eval.base import parse_method_spec
from nbf.eval.metrics import rmse_per_step_batch


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
