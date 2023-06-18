from typing import Callable, Tuple

import numpy as np
from scipy.spatial.distance import cdist

__authors__ = ["Hao Wang"]


class GenerationalDistance:
    def __init__(self, ref: np.ndarray, func: Callable, jac: Callable, hess: Callable, p: float = 2):
        self.ref = ref
        self.p = p
        self.func = func
        self.jac = jac
        self.hess = hess

    def _compute_indices(self, Y: np.ndarray):
        # find for each approximation point, the index of its closest point in the reference set
        self.D = cdist(Y, self.ref, metric="minkowski", p=self.p)
        self.indices = np.argmin(self.D, axis=1)

    def compute(self, X: np.ndarray = None, Y: np.ndarray = None) -> float:
        if Y is None:
            Y = np.array([self.func(x) for x in X])
        self._compute_indices(Y)
        return np.mean(self.D[np.arange(len(Y)), self.indices] ** self.p) ** (1 / self.p)

    def compute_derivatives(
        self, X: np.ndarray, Y: np.ndarray = None, compute_hessian: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: take the p-norm into account
        N = len(X)
        if Y is None:
            Y = np.array([self.func(x) for x in X])
        self._compute_indices(Y)
        # TODO: this part can be parallelized
        J = np.array([self.jac(x) for x in X])  # (N, n_objective, dim)
        diff = Y - self.ref[self.indices]  # (N, n_objective)
        grad = np.einsum("ijk,ij->ik", J, diff)  # (N, dim)
        if compute_hessian:
            H = np.array([self.hess(x) for x in X])  # (N, n_objective, dim, dim)
            hessian = np.einsum("ijk,ijl->ikl", J, J) + np.einsum("ijkl,ij->ikl", H, diff)  # (N, dim, dim)
            return grad * 2 / N, hessian * 2 / N
        else:
            return grad * 2 / N
