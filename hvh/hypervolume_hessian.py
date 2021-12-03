from typing import List, Tuple, Union

import numpy as np

from .hypervolume import hypervolume

# from numba.core.decorators import njit


__author__ = ["Hao Wang"]


def get_non_dominated(pareto_front: np.ndarray, return_index: bool = False):
    """Find pareto front (undominated part) of the input performance data."""
    if len(pareto_front) == 0:
        return np.array([])
    sorted_indices = np.argsort(pareto_front[:, 0])
    pareto_indices = []
    for idx in sorted_indices:
        # check domination relationship
        a = np.all(pareto_front >= pareto_front[idx], axis=1)
        b = np.any(pareto_front > pareto_front[idx], axis=1)
        if not np.any(np.logical_and(a, b)):
            pareto_indices.append(idx)
    pareto_indices = np.array(pareto_indices)
    pareto_front = pareto_front[pareto_indices].copy()
    return pareto_indices if return_index else pareto_front


def hypervolume_improvement(x: np.ndarray, pareto_front: np.ndarray, ref: np.ndarray):
    return hypervolume(np.vstack([x, pareto_front]), ref) - hypervolume(pareto_front, ref)


class HypervolumeHessian:
    """Analytical Hessian matrix of hypervolume"""

    def __init__(
        self,
        pareto_front: Union[np.ndarray, List[List]],
        ref: Union[np.ndarray, List[List]],
        maximization: bool = True,
    ):
        """Compute the hypervolume Hessian matrix

        Parameters
        ----------
        pareto_front : Union[np.ndarray, List[List]]
            the Pareto front
        ref : Union[np.ndarray, List[List]]
            the reference point
        maximization : bool, optional
            whether the MOP is subject to maximization, by default True
        """
        self.maximization = maximization
        self.pareto_front = pareto_front
        self.ref = ref
        self.N, self.dim = self.pareto_front.shape

    @property
    def ref(self):
        return self._ref

    @ref.setter
    def ref(self, r):
        if not isinstance(r, np.ndarray):
            r = np.asarray(r)
        self._ref = r if self.maximization else -1 * r

    @property
    def pareto_front(self):
        return self._pareto_front

    @pareto_front.setter
    def pareto_front(self, pf):
        if not isinstance(pf, np.ndarray):
            pf = np.asarray(pf)
        self._pareto_front = pf if self.maximization else -1 * pf
        pf_indices = get_non_dominated(self._pareto_front, return_index=True)
        self._dominated_indices = set(range(len(self._pareto_front))) - set(pf_indices)

    def project(
        self, axis: int, i: int, pareto_front: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """projecting the pareto front along `axis` with respect to the i-th point"""
        pareto_front = self.pareto_front if pareto_front is None else pareto_front
        x = pareto_front[i, :]
        # projection: drop the `axis`-th dimension
        x_ = np.delete(x, obj=axis)
        ref_ = np.delete(self.ref, obj=axis)
        idx = np.nonzero(self.pareto_front[:, axis] > x[axis])[0]
        pareto_front_ = np.delete(self.pareto_front[idx, :], obj=axis, axis=1)
        if len(pareto_front_) != 0:
            pareto_indices = get_non_dominated(pareto_front_, return_index=True)
            pareto_front_ = pareto_front_[pareto_indices]
            idx = idx[pareto_indices]
        return x_, pareto_front_, ref_, idx

    def compute(self) -> np.ndarray:
        H = np.zeros((self.N * self.dim, self.N * self.dim))
        for i in range(self.N):
            if i in self._dominated_indices:
                continue
            for k in range(self.dim):
                # project along `axis`
                x_, pareto_front_, ref_, proj_idx = self.project(k, i)
                # partial derivatives ∂(∂HV/∂y_k^i)/∂y^i
                # of shape (1, dim), where the k-th element is zero
                X = np.vstack([x_, pareto_front_])
                pareto_indices = get_non_dominated(X, return_index=True)
                idx = np.where(pareto_indices == 0)[0]
                out = self.hypervolume_gradient(X[pareto_indices], ref_)[idx]
                H[i * self.dim : (i + 1) * self.dim, i * self.dim + k] = np.insert(out, k, 0)
                # partial derivatives ∂(∂HV/∂y_k^i)/∂y^{-i}
                # of shape (len(proj_idx), dim), where the k-th element is zero
                # ∂HV/∂y_k^i is the hypervolume improvement of `x_` w.r.t. `pareto_front_`
                out = self.hypervolume_gradient(np.clip(pareto_front_, ref_, x_), ref=ref_)
                # get the dimension of points in `pareto_front_` that are not dominated by `x_`
                idx = pareto_front_ > x_
                out[idx] = 0
                # hypervolume improvement of points in `pareto_front_` decreases ∂HV/∂y_k^i
                out = np.insert(-1.0 * out, k, 0, axis=1)
                for s, j in enumerate(proj_idx):
                    H[j * self.dim : (j + 1) * self.dim, i * self.dim + k] = out[s]
        return H

    def hypervolume_gradient(self, pareto_front: np.ndarray, ref: np.ndarray) -> np.ndarray:
        if len(ref) == 2:  # 2D is a simple case
            return self._2D_hypervolume_gradient(pareto_front, ref)
        # general HV gradient in higher dimensions
        N, dim = pareto_front.shape
        grad = np.zeros((N, dim))
        for i in range(N):
            for k in range(dim):
                x_, pareto_front_, ref_, _ = self.project(k, i, pareto_front)
                grad[i, k] = hypervolume_improvement(x_, pareto_front_, ref_)
        return grad

    def _2D_hypervolume_gradient(self, pareto_front: np.ndarray, ref: np.ndarray) -> np.ndarray:
        N = len(pareto_front)
        gradient = np.zeros((N, 2))
        # sort the pareto front with repsect to y1
        idx = np.argsort(pareto_front[:, 0])
        sorted_pareto_front = pareto_front[idx]

        y1 = sorted_pareto_front[:, 0]
        y2 = sorted_pareto_front[:, 1]
        gradient[idx, 0] = y2 - np.r_[y2[1:], ref[1]]
        gradient[idx, 1] = y1 - np.r_[ref[0], y1[0:-1]]
        return gradient
