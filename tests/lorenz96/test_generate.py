import json
import os

import jax
import numpy as np
import pytest

from nbf.domains.lorenz96.config import DataConfig
from nbf.domains.lorenz96.generate_data import generate_data, load_data, load_norm_stats


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def tiny_cfg():
    return DataConfig(
        K=8,
        num_scenarios=4,
        block_size=2,
        episode_length=3,
        num_pf_particles=256,
        spinup_time=2.0,
        num_val_scenarios=1,
    )


class TestGenerateData:
    def test_shapes_meta_and_stats(self, key, tiny_cfg, tmp_path):
        out_dir = str(tmp_path / 'l96_belief_data_K8')
        generate_data(tiny_cfg, num_samples=16, key=key, out_dir=out_dir)
        data = load_data(str(tmp_path), 8)

        rows = tiny_cfg.episode_length + 1
        total = tiny_cfg.num_scenarios * rows
        assert data['particles_tp1'].shape == (total, 16, 8)
        assert data['particles_tp1'].dtype == np.float16
        assert data['weights_tp1'].shape == (total, 16)
        assert data['meta'].shape == (total, 3)
        assert np.all(np.isfinite(np.asarray(data['particles_tp1'], dtype=np.float32)))

        w = np.asarray(data['weights_tp1'])
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)
        assert np.allclose(w[0], np.ones(16) / 16, atol=1e-6)

        meta = np.asarray(data['meta'])
        for s in range(tiny_cfg.num_scenarios):
            block = meta[s * rows : (s + 1) * rows]
            assert np.allclose(block[:, 0], block[0, 0])
            assert np.allclose(block[:, 1], block[0, 1])
            assert np.allclose(block[:, 2], np.arange(rows))
            F = float(block[0, 0])
            assert (5.0 <= F <= 7.0) or (9.0 <= F <= 11.0)
            assert 0.5 <= float(block[0, 1]) <= 1.5

        stats = load_norm_stats(str(tmp_path), 8)
        assert -1.0 < stats['norm_mean'] < 5.0
        assert 0.5 < stats['norm_std'] < 8.0
        assert os.path.exists(os.path.join(out_dir, 'norm_stats.json'))
        with open(os.path.join(out_dir, 'norm_stats.json')) as f:
            assert set(json.load(f)) == {'norm_mean', 'norm_std'}

    def test_rejects_indivisible_block_size(self, key, tiny_cfg, tmp_path):
        tiny_cfg.num_scenarios = 5
        with pytest.raises(ValueError, match='divisible'):
            generate_data(tiny_cfg, num_samples=16, key=key, out_dir=str(tmp_path / 'x'))
