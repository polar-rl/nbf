import numpy as np


def fmt_val(v: float) -> str:
    if not np.isfinite(v):
        return '--'
    if v == 0.0:
        return '0'
    a = abs(v)
    if 1e-2 <= a < 1e3:
        return f'{v:.3f}'
    return f'{v:.2e}'.replace('e-0', 'e-').replace('e+0', 'e')


def fmt_or_dash(x: float, fmt: str = '.3f') -> str:
    if not np.isfinite(x):
        return '--'
    return format(x, fmt)


def latex_escape(s: str) -> str:
    return s.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
