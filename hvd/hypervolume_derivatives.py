from typing import Dict, List, Tuple, Union

# TODO: remove dependencies on `autograd`
import autograd.numpy as np
import numpy as np
from autograd import hessian, jacobian
from scipy.linalg import block_diag

from .hypervolume import hypervolume
from .utils import non_domin_sort

__author__ = ["Hao Wang"]


def HVY(n_objective: int, ref: np.ndarray) -> callable:
    def _HVY(Y: np.ndarray) -> float:
        return hypervolume(Y.reshape(-1, n_objective), ref)

    return _HVY


def HVX(n_decision_var: int, f: callable, ref: np.ndarray) -> callable:
    def _HVX(X: np.ndarray) -> float:
        return hypervolume(np.array([f(x) for x in X.reshape(-1, n_decision_var)]), ref)

    return _HVX


def hypervolume_improvement(x: np.ndarray, pareto_front: np.ndarray, ref: np.ndarray) -> float:
    """minization is assumed"""
    return hypervolume(np.vstack([x, pareto_front]), ref) - hypervolume(pareto_front.copy(), ref)


class HypervolumeDerivatives:
    """Analytical gradient and Hessian matrix of hypervolume indicator"""

    def __init__(
        self,
        n_decision_var: int,
        n_objective: int,
        ref: Union[np.ndarray, List[List]],
        func: callable = None,
        jac: callable = None,
        hessian: callable = None,
        minimization: bool = True,
    ):
        """Compute the hypervolume Hessian matrix

        Parameters
        ----------
        n_decision_var : int
            dimension of the decision space
        n_objective : int
            dimension of the objective space
        ref : Union[np.ndarray, List[List]]
            the reference point
        func : callable, optional
            the MOP which takes a decision point as input and returns a vector of shape `(n_objective, 1)`,
            by default None, which evaluates to an identity function
        jac : callable, optional
            Jacobian of the MOP which takes a decision point as input and returns a matrix of
            shape `(n_objective, n_decision_var)`, by default None, which evaluates an identity matrix
        hessian : callable, optional
            Hessian matrix of the MOP which takes a decision point as input and returns a tensor of
            shape `(n_objective, n_decision_var, n_decision_var)`, by default None, which evaluates a zero tensor
        maximization : bool, optional
            whether the MOP is subject to maximization, by default True
        """
        if func is None:
            func = lambda x: x
        if jac is None:
            jac = lambda x: np.diag(np.ones(len(x)))
        if hessian is None:
            hessian = lambda x: np.zeros((len(x), len(x), len(x)))

        self.n_decision_var = int(n_decision_var)
        self.n_objective = int(n_objective)
        self.func = func if minimization else lambda x: -1 * func(x)
        self.jac = jac if minimization else lambda x: -1 * jac(x)
        self.hessian = hessian if minimization else lambda x: -1 * hessian(x)
        self.minimization = minimization
        self.ref = ref
        # compute those when needed for the first time
        self._HV_Jac: callable = None
        self._HV_Hessian: callable = None

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
        assert self.n_objective == self._objective_points.shape[1]

        if self._objective_points.shape[1] == 1:
            self._nondominated_indices = np.array([np.argmin(self._objective_points.ravel())])
        else:
            self._nondominated_indices = non_domin_sort(self._objective_points, only_front_indices=True)[0]
        # self._nondominated_indices = get_non_dominated(self._objective_points, return_index=True)
        self._dominated_indices = set(range(len(self._objective_points))) - set(self._nondominated_indices)

    def _check_X(self, X: np.ndarray):
        X = np.atleast_2d(X)
        if X.shape[1] != self.n_decision_var:
            X = X.reshape(-1, self.n_decision_var)
        return X

    def _check_Y(self, Y: np.ndarray):
        Y = np.atleast_2d(Y)
        if Y.shape[1] != self.n_objective:
            Y = Y.reshape(-1, self.n_decision_var)
        return Y

    def HV(self, X: np.ndarray) -> float:
        X = self._check_X(X)
        Y = np.array([self.func(x) for x in X])
        return hypervolume(Y, self.ref)

    def project(
        self, axis: int, i: int, pareto_front: np.ndarray = None, ref: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """projecting the Pareto front along `axis` with respect to the i-th point"""
        pareto_front = self.objective_points if pareto_front is None else pareto_front
        ref = self.ref if ref is None else ref
        y = pareto_front[i, :]
        # projection: drop the `axis`-th dimension
        y_ = np.delete(y, obj=axis)
        ref_ = np.delete(ref, obj=axis)
        idx = np.nonzero(pareto_front[:, axis] < y[axis])[0]
        pareto_front_ = np.delete(pareto_front[idx, :], obj=axis, axis=1)
        if len(pareto_front_) != 0:
            if pareto_front_.shape[1] == 1:
                pareto_indices = np.array([np.argmin(pareto_front_.ravel())])
            else:
                pareto_indices = non_domin_sort(pareto_front_, only_front_indices=True)[0]
            # NOTE: implement a fast algorithm to get the non dominated subset
            # pareto_indices = get_non_dominated(pareto_front_, return_index=True)
            pareto_front_ = pareto_front_[pareto_indices]
            idx = idx[pareto_indices]
        return y_, pareto_front_, ref_, idx

    def _compute_gradient(self, X: np.ndarray, Y: np.ndarray = None) -> Dict[str, np.ndarray]:
        """compute the hypervolume gradient using analytical expressions

        Parameters
        ----------
        X : Union[np.ndarray, List[List]]
            the decision points at which the derivatives are computed
            It should have a shape of (`N`, `n_decision_var`)

        Returns
        -------
        Dict[str, np.ndarray]
            {
                "HVdX": gradient of the hypervolume indicator w.r.t. the decision variable of
                    shape (`N` * `n_decision_var`, ),
                "HVdY": gradient of the hypervolume indicator w.r.t. the objective variable of
                    shape (`N` * `n_objective`, ),
                "YdX": Jacobian of the objective function w.r.t. the decision variable
                    of shape `(N * n_objective, N * n_decision_var)`, a block-diagonal matrix,
                "YdX2": Hessian tensor of the objective function  w.r.t. the objective variable
                    of shape `(N * n_objective, N * n_decision_var, N * n_decision_var)`,
                    a "block-diagonal" tensor,
            }
        """
        X = self._check_X(X)
        Y, YdX, YdX2 = self._compute_objective_derivatives(X, Y)
        self.objective_points = Y
        HVdY = np.zeros((self.N, self.n_objective))
        HVdY[self._nondominated_indices] = self.hypervolume_dY(
            self.objective_points[self._nondominated_indices], self.ref
        )
        HVdY = HVdY.reshape(1, -1)[0]
        HVdX = HVdY @ YdX
        return dict(HVdX=HVdX, HVdY=HVdY, YdX=YdX, YdX2=YdX2)

    def compute(self, X: np.ndarray, Y: np.ndarray = None) -> Dict[str, np.ndarray]:
        """compute the hypervolume gradient and Hessian matrix using analytical expressions

        Parameters
        ----------
        X : Union[np.ndarray, List[List]]
            the decision points at which the derivatives are computed
            It should have a shape of (N, n_decision_var)

        Returns
        -------
        Dict[str, np.ndarray]
            {
                "HVdX": gradient of the hypervolume indicator w.r.t. the decision variable
                    of shape (`N` * `n_decision_var`, ),
                "HVdY": gradient of the hypervolume indicator w.r.t. the objective variable
                    of shape (`N` * `n_objective`, ),
                "HVdX2": Hessian of the hypervolume indicator w.r.t. the decision variable
                    of shape (`N` * `n_decision_var`, `N` * `n_decision_var`),
                "HVdY2": Hessian of the hypervolume indicator w.r.t. the objective variable
                    of shape (`N` * `n_objective`, `N` * `n_objective`)
            }
        """
        X = self._check_X(X)
        res = self._compute_gradient(X, Y)
        HVdY, HVdX, YdX, YdX2 = res["HVdY"], res["HVdX"], res["YdX"], res["YdX2"]
        HVdY2 = np.zeros((self.N * self.n_objective, self.N * self.n_objective))
        for i in range(self.N):
            if i in self._dominated_indices:  # if the point is dominated
                continue
            for k in range(self.n_objective):
                # project along `axis`; `*_` indicates variables in the projected subspace
                y_, pareto_front_, ref_, proj_idx = self.project(axis=k, i=i)
                # partial derivatives ∂(∂HV/∂y_k^i)/∂y^i
                # of shape (1, dim), where the k-th element is zero
                Y_ = np.vstack([y_, pareto_front_])
                pareto_indices = (
                    np.array([np.argmin(Y_.ravel())])
                    if Y_.shape[1] == 1
                    else non_domin_sort(Y_, only_front_indices=True)[0]
                )
                # pareto_indices = get_non_dominated(Y_, return_index=True)
                idx = np.where(pareto_indices == 0)[0]
                out = self.hypervolume_dY(Y_[pareto_indices], ref_)[idx]

                rows = slice(i * self.n_objective, (i + 1) * self.n_objective)
                HVdY2[rows, i * self.n_objective + k] = np.insert(-1.0 * out, k, 0)
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
                    rows = slice(j * self.n_objective, (j + 1) * self.n_objective)
                    HVdY2[rows, i * self.n_objective + k] = out[s]

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

    def compute_automatic_differentiation(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """compute the hypervolume gradient and Hessian matrix using automatic differentiation

        Parameters
        ----------
        X : Union[np.ndarray, List[List]]
            the decision points at which the derivatives are computed
            It should have a shape of (N, n_decision_var)

        Returns
        -------
        Dict[str, np.ndarray]
            {
                "HVdX": gradient of the hypervolume indicator w.r.t. the decision variable
                    of shape (`N` * `n_decision_var`, ),
                "HVdY": gradient of the hypervolume indicator w.r.t. the objective variable
                    of shape (`N` * `n_objective`, ),
                "HVdX2": Hessian of the hypervolume indicator w.r.t. the decision variable
                    of shape (`N` * `n_decision_var`, `N` * `n_decision_var`),
                "HVdY2": Hessian of the hypervolume indicator w.r.t. the objective variable
                    of shape (`N` * `n_objective`, `N` * `n_objective`)
            }
        """
        if self._HV_Jac is None:
            self._HV_Jac = jacobian(HVY(self.n_objective, self.ref))
        if self._HV_Hessian is None:
            self._HV_Hessian = hessian(HVY(self.n_objective, self.ref))

        X = self._check_X(X)
        Y, YdX, YdX2 = self._compute_objective_derivatives(X)
        # NOTE: atuograd does not support matrix input
        HVdY = self._HV_Jac(Y.ravel())
        HVdY2 = self._HV_Hessian(Y.ravel())
        HVdX = HVdY @ YdX
        HVdX2 = YdX.T @ HVdY2 @ YdX + np.einsum("...i,i...", HVdY, YdX2)
        HVdX2 = (HVdX2 + HVdX2.T) / 2
        return dict(HVdX=HVdX, HVdY=HVdY if self.minimization else -1 * HVdY, HVdX2=HVdX2, HVdY2=HVdY2)

    def hypervolume_dY(self, pareto_front: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """compute the gradient of hypervolume indicator in the objective space, i.e.,
        \partial HV / \partial Y

        Args:
            pareto_front (np.ndarray): the Pareto front of shape (n_points, n_objectives)
            ref (np.ndarray): the reference point of shape (n_objective)

        Returns:
            np.ndarray: the hypervolume indicator gradient of shape (n_points, n_objective)
        """
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
        else:
            # higher dimensional cases: recursive computation
            for i in range(N):
                for k in range(dim):
                    y_, pareto_front_, ref_, _ = self.project(k, i, pareto_front, ref)
                    # NOTE: `-1.0` -> since we assume a minimization problem
                    HVdY[i, k] = -1.0 * hypervolume_improvement(y_, pareto_front_, ref_)
        return HVdY

    def _compute_objective_derivatives(
        self, X: np.ndarray, Y: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """compute the objective function value, the Jacobian, and Hessian tensor"""
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.shape[1] != self.n_decision_var:
            X = X.T
        self.N = X.shape[0]  # number of points
        if Y is None:  # do not evaluate the function when `Y` is provided
            Y = np.array([self.func(x) for x in X])  # `(N, n_objective)`
        # Jacobians
        YdX = np.array([self.jac(x) for x in X])  # `(N, n_objective, n_decision_var)`
        YdX = block_diag(*YdX)  # `(N * n_objective, N * n_decision_var)` grouped by points
        # Hessians
        _YdX2 = np.array([self.hessian(x) for x in X])  # `(N, n_objective, n_decision_var, n_decision_var)`
        YdX2 = np.zeros(
            (self.N * self.n_objective, self.N * self.n_decision_var, self.N * self.n_decision_var)
        )
        for i in range(self.N):
            idx = slice(i * self.n_decision_var, (i + 1) * self.n_decision_var)
            YdX2[i * self.n_objective : (i + 1) * self.n_objective, idx, idx] = _YdX2[i, ...]
        return Y, YdX, YdX2
