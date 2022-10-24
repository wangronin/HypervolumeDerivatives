import time
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.linalg import block_diag

from .hypervolume import hypervolume
from .utils import non_domin_sort

__author__ = ["Hao Wang"]


def hypervolume_improvement(x: np.ndarray, pareto_front: np.ndarray, ref: np.ndarray) -> float:
    """minization is assumed"""
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
        minimization: bool = True,
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
        self.func = func if minimization else lambda x: -1 * func(x)
        self.jac = jac if minimization else lambda x: -1 * jac(x)
        self.hessian = hessian if minimization else lambda x: -1 * hessian(x)
        self.minimization = minimization
        self.ref = ref

    @property
    def ref(self):
        return self._ref

    @ref.setter
    def ref(self, r):
        if not isinstance(r, np.ndarray):
            r = np.asarray(r)
        self._ref = r if self.minimization else -1 * r

    @property
    def objective_points(self):
        return self._objective_points

    @objective_points.setter
    def objective_points(self, points):
        if not isinstance(points, np.ndarray):
            points = np.asarray(points)
        self._objective_points = points
        assert self.dim_m == self._objective_points.shape[1]

        if self._objective_points.shape[1] == 1:
            self._nondominated_indices = np.array([np.argmin(self._objective_points.ravel())])
        else:
            self._nondominated_indices = non_domin_sort(self._objective_points, only_front_indices=True)[0]
        # self._nondominated_indices = get_non_dominated(self._objective_points, return_index=True)
        self._dominated_indices = set(range(len(self._objective_points))) - set(self._nondominated_indices)

    def _check_X(self, X: np.ndarray):
        X = np.atleast_2d(X)
        if X.shape[1] != self.dim_d:
            X = X.reshape(-1, self.dim_d)
        return X

    def _check_Y(self, Y: np.ndarray):
        Y = np.atleast_2d(Y)
        if Y.shape[1] != self.dim_m:
            Y = Y.reshape(-1, self.dim_d)
        return Y

    def HV(self, X: np.ndarray) -> float:
        X = self._check_X(X)
        Y = np.array([self.func(x) for x in X])
        return hypervolume(Y, self.ref)

    def project(
        self, axis: int, i: int, pareto_front: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """projecting the Pareto front along `axis` with respect to the i-th point"""
        pareto_front = self.objective_points if pareto_front is None else pareto_front
        y = pareto_front[i, :]
        # projection: drop the `axis`-th dimension
        y_ = np.delete(y, obj=axis)
        ref_ = np.delete(self.ref, obj=axis)
        idx = np.nonzero(self.objective_points[:, axis] < y[axis])[0]
        pareto_front_ = np.delete(self.objective_points[idx, :], obj=axis, axis=1)
        if len(pareto_front_) != 0:
            if pareto_front_.shape[1] == 1:
                pareto_indices = np.array([np.argmin(pareto_front_.ravel())])
            else:
                pareto_indices = non_domin_sort(pareto_front_, only_front_indices=True)[0]
            # pareto_indices = get_non_dominated(pareto_front_, return_index=True)
            pareto_front_ = pareto_front_[pareto_indices]
            idx = idx[pareto_indices]
        return y_, pareto_front_, ref_, idx

    def compute_gradient(self, X: np.ndarray, Y: np.ndarray = None) -> Dict[str, np.ndarray]:
        X = self._check_X(X)
        Y, YdX, YdX2 = self._copmute_objective_derivatives(X, Y)
        self.objective_points = Y
        HVdY = np.zeros((self.N, self.dim_m))
        HVdY[self._nondominated_indices] = self.hypervolume_dY(
            self.objective_points[self._nondominated_indices], self.ref
        )
        HVdY = HVdY.reshape(1, -1)
        HVdX = HVdY @ YdX
        return dict(HVdX=HVdX, HVdY=HVdY, YdX=YdX, YdX2=YdX2)

    def compute_hessian(self, X: np.ndarray, Y: np.ndarray = None) -> Dict[str, np.ndarray]:
        """compute the hypervolume gradient and Hessian matrix

        Parameters
        ----------
        X : Union[np.ndarray, List[List]]
            the decision points at which the derivatives are computed
            It should have a shape of (#number of points, dim_d)

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

        X = self._check_X(X)
        res = self.compute_gradient(X, Y)
        HVdY, HVdX, YdX, YdX2 = res["HVdY"], res["HVdX"], res["YdX"], res["YdX2"]
        HVdY2 = np.zeros((self.N * self.dim_m, self.N * self.dim_m))
        for i in range(self.N):
            if i in self._dominated_indices:  # if the point is dominated
                continue
            for k in range(self.dim_m):
                # project along `axis`; `*_` indicates variables in the projected subspace
                y_, pareto_front_, ref_, proj_idx = self.project(axis=k, i=i)
                # partial derivatives ∂(∂HV/∂y_k^i)/∂y^i
                # of shape (1, dim), where the k-th element is zero
                Y_ = np.vstack([y_, pareto_front_])
                if Y_.shape[1] == 1:
                    pareto_indices = np.array([np.argmin(Y_.ravel())])
                else:
                    pareto_indices = non_domin_sort(Y_, only_front_indices=True)[0]
                # pareto_indices = get_non_dominated(Y_, return_index=True)
                idx = np.where(pareto_indices == 0)[0]
                out = self.hypervolume_dY(Y_[pareto_indices], ref_)[idx]
                HVdY2[i * self.dim_m : (i + 1) * self.dim_m, i * self.dim_m + k] = np.insert(-1.0 * out, k, 0)
                # partial derivatives ∂(∂HV/∂y_k^i)/∂y^{-i}
                # of shape (len(proj_idx), dim), where the k-th element is zero
                # ∂HV/∂y_k^i is the hypervolume improvement of `x_` w.r.t. `pareto_front_`
                out = self.hypervolume_dY(np.clip(pareto_front_, y_, ref_), ref=ref_)
                # get the dimension of points in `pareto_front_` that are not dominated by `x_`
                idx = pareto_front_ < y_
                out[idx] = 0
                # hypervolume improvement of points in `pareto_front_` decreases ∂HV/∂y_k^i
                out = np.insert(out, k, 0, axis=1)
                for s, j in enumerate(proj_idx):
                    HVdY2[j * self.dim_m : (j + 1) * self.dim_m, i * self.dim_m + k] = out[s]

        # TODO: use sparse matrix multiplication here
        HVdX2 = YdX.T @ HVdY2 @ YdX + np.einsum("...i,i...", HVdY, YdX2)
        HVdX2 = (HVdX2 + HVdX2.T) / 2
        return dict(
            Y=self.objective_points if self.minimization else -1 * self.objective_points,
            HVdX=HVdX,
            HVdY=HVdY if self.minimization else -1 * HVdY,
            HVdX2=HVdX2,
            HVdY2=HVdY2,
        )

    def hypervolume_dY(self, pareto_front: np.ndarray, ref: np.ndarray) -> np.ndarray:
        N, dim = pareto_front.shape
        HVdY = np.zeros((N, dim))
        if len(ref) == 1:  # 1D case
            HVdY = np.array([[-1]])
        elif len(ref) == 2:  # 2D case
            # sort the pareto front with repsect to increasing y1 and decreasing y2
            # NOTE: weakly dominbated pointed are handled here
            _, tags = np.unique(pareto_front[:, 0], return_inverse=True)
            idx1 = np.argsort(-tags, kind="stable")[::-1]
            _, tags = np.unique(pareto_front[idx1, 1], return_inverse=True)
            idx2 = np.argsort(-tags, kind="stable")
            idx = idx1[idx2]
            # idx = np.argsort(pareto_front[:, 0])
            sorted_pareto_front = pareto_front[idx]
            y1 = sorted_pareto_front[:, 0]
            y2 = sorted_pareto_front[:, 1]
            HVdY[idx, 0] = y2 - np.r_[ref[1], y2[0:-1]]
            HVdY[idx, 1] = y1 - np.r_[y1[1:], ref[0]]
        else:  # higher dimensional cases
            for i in range(N):
                for k in range(dim):
                    y_, pareto_front_, ref_, _ = self.project(k, i, pareto_front)
                    HVdY[i, k] = -1.0 * hypervolume_improvement(y_, pareto_front_, ref_)
        return HVdY

    def _copmute_objective_derivatives(
        self, X: np.ndarray, Y: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.shape[1] != self.dim_d:
            X = X.T
        self.N = X.shape[0]  # number of points
        # record the CPU time of function evaluations
        t0 = time.process_time_ns()

        if Y is None:  # do not evaluate the function when `Y` is provided
            Y = np.array([self.func(x) for x in X])  # `(N, dim_m)`
        # Jacobians
        YdX = np.array([self.jac(x) for x in X])  # `(N, dim_m, dim_d)`
        YdX = block_diag(*YdX)  # `(N * dim_m, N * dim_d)` grouped by points
        # Hessians
        _YdX2 = np.array([self.hessian(x) for x in X])  # `(N, dim_m, dim_d, dim_d)`
        YdX2 = np.zeros((self.N * self.dim_m, self.N * self.dim_d, self.N * self.dim_d))
        for i in range(self.N):
            idx = slice(i * self.dim_d, (i + 1) * self.dim_d)
            YdX2[i * self.dim_m : (i + 1) * self.dim_m, idx, idx] = _YdX2[i, ...]

        t1 = time.process_time_ns()
        self.FE_CPU_time = t1 - t0
        return Y, YdX, YdX2

    def compute_HVdY_FD(self, Y: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
        """Finite Difference (FD) method for computing the HV gradient

        Args:
            X Union[np.ndarray: the input decision points of shape (N, dim_d)

        Returns:
            np.ndarray:
        """
        grad = np.zeros((self.N, self.dim_m))
        I = np.eye(self.dim_m)
        func = lambda Y: hypervolume(Y, self.ref)
        for i in range(self.N):
            for k in range(self.dim_m):
                Y_minus = Y.copy()
                Y_minus[i] -= epsilon * I[k]
                Y_plus = Y.copy()
                Y_plus[i] += epsilon * I[k]
                grad[i, k] = (func(Y_plus) - func(Y_minus)) / (2 * epsilon)
        return grad

    def compute_HVdY2_FD(self, Y: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
        N = self.N * self.dim_m
        Y_ = Y.reshape(N, -1).ravel()
        H = np.zeros((N, N))
        I = np.eye(N)
        f = lambda A: hypervolume(A.reshape(self.N, -1), self.ref)
        for i in range(N):
            for j in range(N):
                f1 = f(Y_ + epsilon * I[i] + epsilon * I[j])
                f2 = f(Y_ + epsilon * I[i] - epsilon * I[j])
                f3 = f(Y_ - epsilon * I[i] + epsilon * I[j])
                f4 = f(Y_ - epsilon * I[i] - epsilon * I[j])
                numdiff = (f1 - f2 - f3 + f4) / (4 * epsilon**2)
                H[i, j] = numdiff
        return (H + H.T) / 2

    def compute_HVdX_FD(self, X: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
        """Finite Difference (FD) method for computing the HV gradient

        Args:
            X Union[np.ndarray: the input decision points of shape (N, dim_d)

        Returns:
            np.ndarray:
        """
        grad = np.zeros((self.N, self.dim_d))
        I = np.eye(self.dim_d)
        func = lambda X: hypervolume(np.array([self.func(x) for x in X]), self.ref)
        for i in range(self.N):
            for k in range(self.dim_d):
                X_minus = X.copy()
                X_minus[i] -= epsilon * I[k]
                X_plus = X.copy()
                X_plus[i] += epsilon * I[k]
                grad[i, k] = (func(X_plus) - func(X_minus)) / (2 * epsilon)
        return grad

    def compute_HVdX2_FD(self, X: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
        N = self.N * self.dim_d
        X_ = X.reshape(N, -1).ravel()
        H = np.zeros((N, N))
        I = np.eye(N)
        f = lambda A: hypervolume(np.array([self.func(x) for x in A.reshape(self.N, -1)]), self.ref)
        for i in range(N):
            for j in range(N):
                f1 = f(X_ + epsilon * I[i] + epsilon * I[j])
                f2 = f(X_ + epsilon * I[i] - epsilon * I[j])
                f3 = f(X_ - epsilon * I[i] + epsilon * I[j])
                f4 = f(X_ - epsilon * I[i] - epsilon * I[j])
                numdiff = (f1 - f2 - f3 + f4) / (4 * epsilon**2)
                H[i, j] = numdiff
        return (H + H.T) / 2
