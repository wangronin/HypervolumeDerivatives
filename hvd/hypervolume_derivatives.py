from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.linalg import block_diag

from .hypervolume import hypervolume

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
    x = -1 * x.copy()
    pareto_front = -1 * pareto_front.copy()
    ref = -1 * ref.copy()
    return hypervolume(np.vstack([x, pareto_front]), ref) - hypervolume(pareto_front.copy(), ref)


class HypervolumeDerivatives:
    """Analytical gradient and Hessian matrix of hypervolume indicator"""

    def __init__(
        self,
        dim_d: int,
        dim_m: int,
        ref: Union[np.ndarray, List[List]],
        func: callable = None,
        jac: callable = None,
        hessian: callable = None,
        maximization: bool = True,
    ):
        """Compute the hypervolume Hessian matrix

        Parameters
        ----------
        dim_d : int
            dimension of the decision space
        dim_m : int
            dimension of the objective space
        ref : Union[np.ndarray, List[List]]
            the reference point
        func : callable, optional
            the MOP which takes a decision point as input and returns a vector of shape `(dim_m, 1)`,
            by default None, which evaluates to an identity function
        jac : callable, optional
            Jacobian of the MOP which takes a decision point as input and returns a matrix of
            shape `(dim_m, dim_d)`, by default None, which evaluates an identity matrix
        hessian : callable, optional
            Hessian matrix of the MOP which takes a decision point as input and returns a tensor of
            shape `(dim_m, dim_d, dim_d)`, by default None, which evaluates a zero tensor
        maximization : bool, optional
            whether the MOP is subject to maximization, by default True
        """
        if func is None:
            func = lambda x: x
        if jac is None:
            jac = lambda x: np.diag(np.ones(len(x)))
        if hessian is None:
            hessian = lambda x: np.zeros((len(x), len(x), len(x)))

        self.dim_d = int(dim_d)
        self.dim_m = int(dim_m)
        self.func = func if maximization else lambda x: -1 * func(x)
        self.jac = jac if maximization else lambda x: -1 * jac(x)
        self.hessian = hessian if maximization else lambda x: -1 * hessian(x)
        self.maximization = maximization
        self.ref = ref

    @property
    def ref(self):
        return self._ref

    @ref.setter
    def ref(self, r):
        if not isinstance(r, np.ndarray):
            r = np.asarray(r)
        self._ref = r if self.maximization else -1 * r

    @property
    def objective_points(self):
        return self._objective_points

    @objective_points.setter
    def objective_points(self, points):
        if not isinstance(points, np.ndarray):
            points = np.asarray(points)
        self._objective_points = points
        assert self.dim_m == self._objective_points.shape[1]
        self._nondominated_indices = get_non_dominated(self._objective_points, return_index=True)
        self._dominated_indices = set(range(len(self._objective_points))) - set(self._nondominated_indices)

    def project(
        self, axis: int, i: int, pareto_front: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """projecting the Pareto front along `axis` with respect to the i-th point"""
        pareto_front = self.objective_points if pareto_front is None else pareto_front
        y = pareto_front[i, :]
        # projection: drop the `axis`-th dimension
        y_ = np.delete(y, obj=axis)
        ref_ = np.delete(self.ref, obj=axis)
        idx = np.nonzero(self.objective_points[:, axis] > y[axis])[0]
        pareto_front_ = np.delete(self.objective_points[idx, :], obj=axis, axis=1)
        if len(pareto_front_) != 0:
            pareto_indices = get_non_dominated(pareto_front_, return_index=True)
            pareto_front_ = pareto_front_[pareto_indices]
            idx = idx[pareto_indices]
        return y_, pareto_front_, ref_, idx

    def compute(self, X: Union[np.ndarray, List[List]]) -> Dict[str, np.ndarray]:
        """compute the hypervolume gradient and Hessian matrix

        Parameters
        ----------
        X : Union[np.ndarray, List[List]]
            the decision points at which the derivatives are computed

        Returns
        -------
        Dict[str, np.ndarray]
            {
                "HVdX": gradient w.r.t. the decision variable of shape (1, `N` * `dim_d`),
                "HVdY": gradient w.r.t. the objective variable of shape (1, `N` * `dim_m`),
                "HVdX2": Hessian w.r.t. the decision variable of shape (`N` * `dim_d`, `N` * `dim_d`),
                "HVdY2": Hessian w.r.t. the objective variable of shape (`N` * `dim_m`, `N` * `dim_m`)
            }
        """
        Y, YdX, YdX2 = self._copmute_objective_derivatives(X)
        self.objective_points = Y
        HVdY2 = np.zeros((self.N * self.dim_m, self.N * self.dim_m))

        for i in range(self.N):
            if i in self._dominated_indices:  # if the point is dominated
                continue
            for k in range(self.dim_m):
                # project along `axis`; `*_` indicates variables in the projected subspace
                y_, pareto_front_, ref_, proj_idx = self.project(k, i)
                # partial derivatives ∂(∂HV/∂y_k^i)/∂y^i
                # of shape (1, dim), where the k-th element is zero
                Y_ = np.vstack([y_, pareto_front_])
                pareto_indices = get_non_dominated(Y_, return_index=True)
                idx = np.where(pareto_indices == 0)[0]
                out = self.hypervolume_dY(Y_[pareto_indices], ref_)[idx]
                HVdY2[i * self.dim_m : (i + 1) * self.dim_m, i * self.dim_m + k] = np.insert(out, k, 0)
                # partial derivatives ∂(∂HV/∂y_k^i)/∂y^{-i}
                # of shape (len(proj_idx), dim), where the k-th element is zero
                # ∂HV/∂y_k^i is the hypervolume improvement of `x_` w.r.t. `pareto_front_`
                out = self.hypervolume_dY(np.clip(pareto_front_, ref_, y_), ref=ref_)
                # get the dimension of points in `pareto_front_` that are not dominated by `x_`
                idx = pareto_front_ > y_
                out[idx] = 0
                # hypervolume improvement of points in `pareto_front_` decreases ∂HV/∂y_k^i
                out = np.insert(-1.0 * out, k, 0, axis=1)
                for s, j in enumerate(proj_idx):
                    HVdY2[j * self.dim_m : (j + 1) * self.dim_m, i * self.dim_m + k] = out[s]

        HVdY = self.hypervolume_dY(self.objective_points, self.ref).reshape(1, -1)
        HVdX = HVdY @ YdX
        if not self.maximization:
            HVdY *= -1
        # TODO: use sparse matrix multiplication here
        HVdX2 = YdX.T @ HVdY2 @ YdX + np.einsum("...i,i...", HVdY, YdX2)
        return dict(HVdX=HVdX, HVdY=HVdY, HVdX2=HVdX2, HVdY2=HVdY2)

    def hypervolume_dY(self, pareto_front: np.ndarray, ref: np.ndarray) -> np.ndarray:
        N, dim = pareto_front.shape
        HVdY = np.zeros((N, dim))
        if len(ref) == 1:  # 1D case
            HVdY = np.array([[1]])
        elif len(ref) == 2:  # 2D case
            N = len(pareto_front)
            # sort the pareto front with repsect to y1
            idx = np.argsort(pareto_front[:, 0])
            sorted_pareto_front = pareto_front[idx]
            y1 = sorted_pareto_front[:, 0]
            y2 = sorted_pareto_front[:, 1]
            HVdY[idx, 0] = y2 - np.r_[y2[1:], ref[1]]
            HVdY[idx, 1] = y1 - np.r_[ref[0], y1[0:-1]]
        else:  # higher dimensional cases
            for i in range(N):
                for k in range(dim):
                    y_, pareto_front_, ref_, _ = self.project(k, i, pareto_front)
                    HVdY[i, k] = hypervolume_improvement(y_, pareto_front_, ref_)
        return HVdY

    def _copmute_objective_derivatives(
        self, X: Union[np.ndarray, List[List]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.shape[1] != self.dim_d:
            X = X.T
        self.N = X.shape[0]  # number of points
        Y = np.array([self.func(x) for x in X])  # `(N, dim_m)`
        # Jacobians
        YdX = np.array([self.jac(x) for x in X])  # `(N, dim_m, dim_d)`
        YdX = block_diag(*YdX)  # `(N * dim_m, N * dim_d)`
        # Hessians
        _YdX2 = np.array([self.hessian(x) for x in X])  # `(N, dim_m, dim_d, dim_d)`
        YdX2 = np.zeros((self.N * self.dim_m, self.N * self.dim_d, self.N * self.dim_d))
        for i in range(self.N):
            idx = slice(i * self.dim_d, (i + 1) * self.dim_d)
            YdX2[i * self.dim_m : (i + 1) * self.dim_m, idx, idx] = _YdX2[i, ...]
        return Y, YdX, YdX2
