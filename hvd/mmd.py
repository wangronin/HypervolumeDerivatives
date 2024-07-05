import os
from functools import partial
from typing import Dict, List, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jit
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist

from .reference_set import ClusteredReferenceSet

# enable double-precision of JAX
os.environ["JAX_ENABLE_X64"] = "True"

__author__ = "Hao Wang"


@jit
def rational_quadratic(x: np.ndarray, y: np.ndarray, theta: float = 1.0, alpha: float = 1.0) -> float:
    return (jnp.sum((x - y) ** 2) * theta / (2 * alpha) + 1) ** (-alpha)


@jit
def rbf(x: np.ndarray, y: np.ndarray, theta: float = 1.0) -> float:
    return jnp.exp(-theta * jnp.sum((x - y) ** 2))


@jit
def laplace(x: np.ndarray, y: np.ndarray, theta: float = 1.0) -> float:
    return jnp.exp(-theta * jnp.sum((jnp.abs(x - y))))


class MMD:
    """Maximum Mean Discrepancy (MMD) indicator"""

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        ref: np.ndarray,
        func: callable = None,
        jac: callable = None,
        hessian: callable = None,
        kernel: callable = rational_quadratic,
        theta: float = 1.0,
    ) -> None:
        """Maximum Mean Discrepancy (MMD) indicator for multi-objective optimization

        Args:
            n_decision_var (int): the number of decision variables
            n_objective (int): the number of objective functions
            ref (np.ndarray): the reference set of shape (`N`, `n_objective`)
            func (callable, optional): the objective function. Defaults to None.
            jac (callable, optional): the Jacobian of the objective function. Defaults to None.
            hessian (callable, optional): the Hessian of the objective function. Defaults to None.
            kernel (callable, optional): the kernel function. Defaults to `rational_quadratic`.
            theta (float, optional): length-scale of the kernel. Defaults to 1.0.
        """
        self.func = func if func is not None else lambda x: x
        self.jac = jac if jac is not None else lambda x: np.diag(np.ones(len(x)))
        self.hessian = hessian if hessian is not None else lambda x: np.zeros((len(x), len(x), len(x)))
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        self.theta: float = theta  # kernel's length-scale
        self.ref = ref
        self.N = self.ref.shape[0]
        # kernel for correlations between `ref` and `Y`
        self.k = partial(kernel, theta=self.theta)
        self.k_dx = jit(jacrev(self.k))
        self.k_dx2 = jit(jacfwd(jacrev(self.k)))
        self.k_dxdy = jit(jacfwd(jacrev(self.k), argnums=1))  # cross second-order derivatives of the kernel

    def compute(self, X: np.ndarray = None, Y: np.ndarray = None) -> float:
        """Compute the indicator value

        Args:
            X (np.ndarray, optional): the decision points of shape (N, dim). Defaults to None.
            Y (np.ndarray): the Pareto front approximate set of shape (`N`, `self.n_objective`).
                Defaults to None.

        Returns:
            float: MMD value between `Y` and `self.ref`
        """
        if Y is None:
            assert X is not None
            assert self.func is not None
            Y = np.array([self.func(x) for x in X])
        RR = cdist(self.ref, self.ref, metric=self.k)
        YY = cdist(Y, Y, metric=self.k)
        RY = cdist(self.ref, Y, metric=self.k)
        return RR.mean() + YY.mean() - 2 * RY.mean()

    def compute_gradient(self, X: np.ndarray, Y: np.ndarray = None) -> Dict[str, np.ndarray]:
        """compute the gradient of the MMD indicator w.r.t. objective points

        Args:
            X (np.ndarray): the approximation set of shape (`N`, `self.n_decision_var`)

        Returns:
            np.ndarray: the gradient of shape (`N`, `self.n_objective`)
        """
        X = self._check_X(X)
        Y, YdX, YdX2 = self._compute_objective_derivatives(X, Y)
        N = Y.shape[0]
        MMDdY = np.zeros((N, self.n_obj))
        for l, y in enumerate(Y):
            # TODO: `term1` is only correct for stationary kernels
            term1 = np.sum([self.k_dx(y, Y[i]) for i in range(N) if i != l], axis=0)
            term2 = np.sum([self.k_dx(y, self.ref[i]) for i in range(self.N)], axis=0)
            MMDdY[l] = 2 * (term1 / N**2 - term2 / (N * self.N))
        MMDdX = np.einsum("ij,ijk->ik", MMDdY, YdX)
        return dict(MMDdX=MMDdX, MMDdY=MMDdY, Y=Y, YdX=YdX, YdX2=YdX2)

    def compute_hessian(self, X: np.ndarray, Y: np.ndarray = None) -> Dict[str, np.ndarray]:
        """compute the Hessian of the MMD indicator w.r.t. objective points

        Args:
            Y (np.ndarray): the approximation set of shape (`N`, `self.n_objective`)

        Returns:
            np.ndarray: the Hessian of shape (`N * self.n_objective`, `N * self.n_objective`)
        """
        out = self.compute_gradient(X, Y)
        Y, YdX, YdX2, MMDdY, MMDdX = out["Y"], out["YdX"], out["YdX2"], out["MMDdY"], out["MMDdX"]
        N, dim_y = Y.shape
        dim_x = self.n_var
        MMDdY2 = np.zeros((N * dim_y, N * dim_y))
        MMDdX2 = np.zeros((N * dim_x, N * dim_x))
        for l in range(N):
            for m in range(l, N):
                # compute MMDdY2
                r, c = slice(m * dim_y, (m + 1) * dim_y), slice(l * dim_y, (l + 1) * dim_y)
                if m != l:
                    MMDdY2[r, c] = 2 * self.k_dxdy(Y[l], Y[m]) / N**2
                    MMDdY2[c, r] = MMDdY2[r, c].T
                else:
                    # TODO: `term1` is only correct for stationary kernels
                    term1 = np.sum([self.k_dx2(Y[l], Y[i]) for i in range(N) if i != l], axis=0)
                    term2 = np.sum([self.k_dx2(Y[l], self.ref[i]) for i in range(self.N)], axis=0)
                    MMDdY2[r, c] = 2 * (term1 / N**2 - term2 / (N * self.N))
                # compute MMDdX2
                rr, cc = slice(m * dim_x, (m + 1) * dim_x), slice(l * dim_x, (l + 1) * dim_x)
                MMDdX2[rr, cc] = YdX[m].T @ MMDdY2[r, c] @ YdX[l]
                MMDdX2[cc, rr] = MMDdX2[rr, cc].T
        MMDdX2 += block_diag(*np.einsum("ij,ij...->i...", MMDdY, YdX2))
        return dict(MMDdX2=MMDdX2, MMDdY2=MMDdY2, MMDdX=MMDdX, MMDdY=MMDdY)

    def _compute_objective_derivatives(
        self, X: np.ndarray, Y: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """compute the objective function value, the Jacobian, and Hessian tensor"""
        if Y is None:
            Y = np.array([self.func(x) for x in X])  # `(N, n_objective)`
        assert Y.shape[1] == self.n_obj
        # Jacobians of the objective function
        YdX = np.array([self.jac(x) for x in X])  # `(N, n_objective, n_decision_var)`
        # Hessians of the objective function
        YdX2 = np.array([self.hessian(x) for x in X])  # `(N, n_objective, n_decision_var, n_decision_var)`
        return Y, YdX, YdX2

    def _check_X(self, X: Union[np.ndarray, List]) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.shape[1] != self.n_var:
            X = X.T
        return X


class MMDMatching:
    """Maximum Mean Discrepancy (MMD) indicator with matching between approximation and reference points"""

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        ref: Union[np.ndarray, ClusteredReferenceSet],
        func: callable = None,
        jac: callable = None,
        hessian: callable = None,
        kernel: callable = rbf,
        theta: float = 1.0,
        beta: float = 0.5,
    ) -> None:
        """Maximum Mean Discrepancy (MMD) indicator for multi-objective optimization

        Args:
            n_var (int): the number of decision variables
            n_obj (int): the number of objective functions
            ref (np.ndarray): the reference set of shape (`N`, `n_objective`)
            func (callable, optional): the objective function. Defaults to None.
            jac (callable, optional): the Jacobian of the objective function. Defaults to None.
            hessian (callable, optional): the Hessian of the objective function. Defaults to None.
            kernel (callable, optional): the kernel function. Defaults to `rational_quadratic`.
            theta (float, optional): length-scale of the kernel. Defaults to 1.0.
            beta (float, optional): coefficient to scale the RKHS norm of the approximation set.
                Defaults to 0.5.
        """
        if isinstance(ref, np.ndarray):
            ref = ClusteredReferenceSet(ref)
        self.func = func if func is not None else lambda x: x
        self.jac = jac if jac is not None else lambda x: np.diag(np.ones(len(x)))
        self.hessian = hessian if hessian is not None else lambda x: np.zeros((len(x), len(x), len(x)))
        self.n_decision_var = int(n_var)
        self.n_objective = int(n_obj)
        self.theta: float = theta  # kernel's length-scale
        self.ref = ref
        self.N = self.ref.N
        self.k = partial(kernel, theta=self.theta)
        self.k_dx = jit(jacrev(self.k))
        self.k_dx2 = jit(jacfwd(jacrev(self.k)))
        self.k_dxdy = jit(jacfwd(jacrev(self.k), argnums=1))  # cross second-order derivatives of the kernel
        self.beta = beta  # the weight to scale the spread term, RKHS norm of images of `Y`

    def compute(self, X: np.ndarray = None, Y: np.ndarray = None) -> float:
        """Compute the indicator value

        Args:
            X (np.ndarray, optional): the decision points of shape (N, dim). Defaults to None.
            Y (np.ndarray): the Pareto front approximate set of shape (`N`, `self.n_objective`).
                Defaults to None.

        Returns:
            float: MMD value between `Y` and `self.ref`
        """
        if Y is None:
            assert X is not None
            assert self.func is not None
            Y = np.array([self.func(x) for x in X])
        self.ref.match(Y)
        return (
            self.beta * cdist(Y, Y, metric=self.k).mean()
            + np.array([2 - 2 * self.k(y, self.ref.medoids[i]) for i, y in enumerate(Y)]).mean()
        )

    def compute_derivatives(
        self,
        X: np.ndarray,
        Y: np.ndarray = None,
        compute_hessian: bool = True,
        jacobian: np.ndarray = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """compute the derivatives of the inverted generational distance^p

        Args:
            X (np.ndarray): the decision points of shape (N, dim).
            Y (np.ndarray, optional): the objective points of shape (N, n_objective). Defaults to None.
            compute_hessian (bool, optional): whether the Hessian is computed. Defaults to True.
            jacobian (np.ndarray, optional): Jacobian of the objective function at `X`. Defaults to None.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                if `compute_hessian` = True, it returns (gradient, Hessian)
                otherwise, it returns (gradient, )
        """
        if compute_hessian:
            out = self.compute_hessian(X, Y)
            grad, hessian = out["MMDdX"], out["MMDdX2"]
        else:
            grad = self.compute_gradient(X, Y, jacobian)["MMDdX"]
        return (grad, hessian) if compute_hessian else grad

    def compute_gradient(
        self, X: np.ndarray, Y: np.ndarray = None, jacobian: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """compute the gradient of the MMD indicator w.r.t. objective points

        Args:
            X (np.ndarray): the approximation set of shape (`N`, `self.n_decision_var`)

        Returns:
            np.ndarray: the gradient of shape (`N`, `self.n_objective`)
        """
        X = self._check_X(X)
        Y, YdX, YdX2 = self._compute_objective_derivatives(X, Y, jacobian)
        self.ref.match(Y)  # matching `Y` to the medoids of the reference set
        N = Y.shape[0]
        MMDdY = np.zeros((N, self.n_objective))
        for l, y in enumerate(Y):
            # TODO: `term1` is only correct for stationary kernels
            term1 = 2 * self.beta * np.sum([self.k_dx(y, Y[i]) for i in range(N) if i != l], axis=0) / N**2
            term2 = 2 * self.k_dx(y, self.ref.medoids[l]) / N
            MMDdY[l] = term1 - term2
        MMDdX = np.einsum("ijk,ij->ik", YdX, MMDdY)
        return dict(MMDdX=MMDdX, MMDdY=MMDdY, Y=Y, YdX=YdX, YdX2=YdX2)

    def compute_hessian(self, X: np.ndarray, Y: np.ndarray = None) -> Dict[str, np.ndarray]:
        """compute the Hessian of the MMD indicator w.r.t. objective points

        Args:
            Y (np.ndarray): the approximation set of shape (`N`, `self.n_objective`)

        Returns:
            np.ndarray: the Hessian of shape (`N * self.n_objective`, `N * self.n_objective`)
        """
        out = self.compute_gradient(X, Y)
        Y, YdX, YdX2, MMDdY, MMDdX = out["Y"], out["YdX"], out["YdX2"], out["MMDdY"], out["MMDdX"]
        N, dim_y = Y.shape
        dim_x = self.n_decision_var
        MMDdY2 = np.zeros((N * dim_y, N * dim_y))
        MMDdX2 = np.zeros((N * dim_x, N * dim_x))
        for l in range(N):
            for m in range(l, N):
                # compute MMDdY2
                r, c = slice(m * dim_y, (m + 1) * dim_y), slice(l * dim_y, (l + 1) * dim_y)
                if m != l:
                    MMDdY2[r, c] = 2 * self.beta * self.k_dxdy(Y[l], Y[m]) / N**2
                    MMDdY2[c, r] = MMDdY2[r, c].T
                else:
                    # TODO: `term1` is only correct for stationary kernels
                    term1 = np.sum([self.k_dx2(Y[l], Y[i]) for i in range(N) if i != l], axis=0) / N**2
                    term2 = self.k_dx2(Y[l], self.ref.medoids[l]) / N
                    MMDdY2[r, c] = 2 * (self.beta * term1 - term2)
                # compute MMDdX2
                rr, cc = slice(m * dim_x, (m + 1) * dim_x), slice(l * dim_x, (l + 1) * dim_x)
                MMDdX2[rr, cc] = YdX[m].T @ MMDdY2[r, c] @ YdX[l]
                MMDdX2[cc, rr] = MMDdX2[rr, cc].T
        MMDdX2 += block_diag(*np.einsum("ij,ij...->i...", MMDdY, YdX2))
        return dict(MMDdX2=MMDdX2, MMDdY2=MMDdY2, MMDdX=MMDdX, MMDdY=MMDdY)

    def _compute_objective_derivatives(
        self, X: np.ndarray, Y: np.ndarray = None, jacobian: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """compute the objective function value, the Jacobian, and Hessian tensor"""
        if Y is None:
            Y = np.array([self.func(x) for x in X])  # `(N, n_objective)`
        assert Y.shape[1] == self.n_objective
        # Jacobians of the objective function
        #  of shape `(N, n_objective, n_decision_var)`
        YdX = np.array([self.jac(x) for x in X]) if jacobian is None else jacobian
        # Hessians of the objective function
        YdX2 = np.array([self.hessian(x) for x in X])  # `(N, n_objective, n_decision_var, n_decision_var)`
        return Y, YdX, YdX2

    def _check_X(self, X: Union[np.ndarray, List]) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.shape[1] != self.n_decision_var:
            X = X.T
        return X
