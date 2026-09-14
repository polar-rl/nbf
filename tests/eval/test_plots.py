import numpy as np
import pytest

from nbf.eval.plots import (
    load_cell,
    order_specs,
    plot_compute_matched,
    plot_per_step,
    plot_sweep,
    series_style,
)


def _fake_cell_npz(path, subcells, specs, N=3, T=4, seeds=(0, 1)):
    rng = np.random.default_rng(0)
    payload = {}
    for sc in subcells:
        for seed in seeds:
            for spec in specs:
                for metric in ('rmse', 'crps'):
                    payload[f'{sc}|s{seed}|{spec}|{metric}'] = rng.random((N, T)).astype(np.float32)
                if spec.startswith('nbf'):
                    payload[f'{sc}|s{seed}|{spec}|nll'] = rng.random((N, T)).astype(np.float32)
                payload[f'{sc}|s{seed}|{spec}|walltime_s'] = np.asarray(0.5 + rng.random())
    np.savez(path, **payload)


def test_series_style_and_order():
    assert series_style('nbf_2048')[0] == 'NBF(2048)'
    assert series_style('pf_reg_8192')[0] == 'PF-reg(8192)'
    assert order_specs(['pf_8192', 'nbf_256', 'enkf_50', 'pf_op_1024', 'nbf_1024']) == [
        'nbf_256',
        'nbf_1024',
        'pf_op_1024',
        'pf_8192',
        'enkf_50',
    ]


def test_load_and_plots(tmp_path):
    npz = tmp_path / 'l2.npz'
    _fake_cell_npz(npz, ['f8'], ['nbf_256', 'pf_1024', 'enkf_20'])
    cell = load_cell(str(npz))
    assert cell['f8']['nbf_256']['rmse'].shape == (6, 4)  # 2 seeds x 3 episodes pooled
    assert cell['f8']['pf_1024']['walltime_s'].shape == (2,)
    plot_per_step(cell, 'f8', ['nll', 'rmse', 'crps'], str(tmp_path / 'ps.pdf'))
    plot_compute_matched(cell, 'f8', 'crps', str(tmp_path / 'cm.pdf'))
    assert (tmp_path / 'ps.pdf').exists() and (tmp_path / 'cm.pdf').exists()


@pytest.mark.parametrize('subcells', [['sig0p1', 'sig1', 'sig3'], ['wall_hugger', 'evasive']])
def test_sweep_numeric_and_categorical(tmp_path, subcells):
    npz = tmp_path / 'c.npz'
    _fake_cell_npz(npz, subcells, ['nbf_256', 'pf_1024'])
    plot_sweep(load_cell(str(npz)), 'rmse', str(tmp_path / 'sw.pdf'))
    assert (tmp_path / 'sw.pdf').exists()
