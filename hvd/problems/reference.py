import math
from collections.abc import Iterator
import numpy as np


def get_ref_dirs(n_obj: int, n_points: int | None = None, n_partitions: int | None = None) -> np.ndarray:
    """Das--Dennis simplex directions without a pymoo dependency."""
    if n_partitions is None:
        target = n_points or (100 if n_obj == 2 else 496)
        n_partitions = 1
        while math.comb(n_partitions + n_obj - 1, n_obj - 1) < target:
            n_partitions += 1
    def compositions(
        total: int, parts: int, prefix: tuple[int, ...] = ()
    ) -> Iterator[tuple[int, ...]]:
        if parts == 1:
            yield prefix + (total,)
            return
        for value in range(total + 1):
            yield from compositions(total - value, parts - 1, prefix + (value,))

    return np.asarray(list(compositions(n_partitions, n_obj)), dtype=float) / n_partitions


def generic_sphere(ref_dirs: np.ndarray) -> np.ndarray:
    return ref_dirs / np.tile(np.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))
