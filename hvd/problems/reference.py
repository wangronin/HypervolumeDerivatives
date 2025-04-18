import numpy as np
from pymoo.util.reference_direction import UniformReferenceDirectionFactory


def get_ref_dirs(n_obj: int) -> np.ndarray:
    if n_obj == 2:
        ref_dirs = UniformReferenceDirectionFactory(2, n_points=100).do()
    elif n_obj == 3:
        ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=23).do()
    else:
        raise Exception("Please provide reference directions for more than 3 objectives!")
    return ref_dirs


def generic_sphere(ref_dirs):
    return ref_dirs / np.tile(np.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))
