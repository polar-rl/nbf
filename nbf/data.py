import os

import numpy as np


def allocate_memmaps(out_dir: str, specs: dict[str, tuple[np.dtype, tuple[int, ...]]]):
    os.makedirs(out_dir, exist_ok=True)
    return {
        name: np.lib.format.open_memmap(
            os.path.join(out_dir, f'{name}.npy'), mode='w+', dtype=dtype, shape=shape
        )
        for name, (dtype, shape) in specs.items()
    }


def load_arrays(dpath: str, names: tuple[str, ...], mmap_mode: str = 'r') -> dict[str, np.ndarray]:
    return {
        name: np.load(os.path.join(dpath, f'{name}.npy'), mmap_mode=mmap_mode) for name in names
    }
