from typing import Callable, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist

from .reference_set import ReferenceSet

__authors__ = ["Hao Wang"]


class GenerationalDistance:
    def __init__(
        self,
        ref: Union[np.ndarray, ReferenceSet],
        func: Callable = None,
        jac: Callable = None,
        hess: Callable = None,
        p: float = 2,
    ):
        """Generational Distance

        Args:
            ref (np.ndarray): the reference set, of shape (n_point, n_objective)
            func (Callable): the objective function, which returns an array of shape (n_objective, )
            jac (Callable): the Jacobian function, which returns an array of shape (n_objective, dim)
            hess (Callable): the Hessian function, which returns an array of shape (n_objective, dim, dim)
            p (float, optional): parameter in the p-norm. Defaults to 2.
        """
        if isinstance(ref, np.ndarray):
            ref = ReferenceSet(ref)
        self.ref = ref
        self.p = p
        self.func = func
        self.jac = jac
        self.hess = hess

    def __str__(self) -> str:
        return "GD"

    def _compute_indices(self, Y: np.ndarray):
        # find for each approximation point, the index of its closest point in the reference set
        self.D = cdist(Y, self.ref.reference_set, metric="minkowski")
        self.indices = np.argmin(self.D, axis=1)

    def compute(self, X: np.ndarray = None, Y: np.ndarray = None) -> float:
        """compute the generational distance value

        Args:
            X (np.ndarray, optional): the decision points of shape (N, dim). Defaults to None.
            Y (np.ndarray, optional): the objective points of shape (N, n_objective). Defaults to None.
                `X` and `Y` cannot be None at the same time

        Returns:
            float: the indicator value
        """
        if Y is None:
            assert X is not None
            Y = np.array([self.func(x) for x in X])
        self._compute_indices(Y)
        return np.mean(self.D[np.arange(len(Y)), self.indices] ** self.p) ** (1 / self.p)

    def compute_derivatives(
        self,
        X: np.ndarray,
        Y: np.ndarray = None,
        compute_hessian: bool = True,
        Jacobian: np.ndarray = None,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """compute the derivatives of the generational distance^p

        Args:
            X (np.ndarray): the decision points of shape (N, dim).
            Y (np.ndarray, optional): the objective points of shape (N, n_objective). Defaults to None.
            compute_hessian (bool, optional): whether the Hessian is computed. Defaults to True.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                if `compute_hessian` = True, it returns (gradient, Hessian)
                otherwise, it returns (gradient, )
        """
        N, dim = X.shape
        c1 = self.p / N
        if Y is None:
            Y = np.array([self.func(x) for x in X])

        self._compute_indices(Y)
        # Jacobian of the objective function of shape (N, n_objective, dim)
        J = np.array([self.jac(x) for x in X]) if Jacobian is None else Jacobian
        diff = Y - self.ref.reference_set[self.indices]  # (N, n_objective)
        diff_norm = np.sqrt(np.sum(diff**2, axis=1)).reshape(-1, 1)  # (N, 1)
        grad_ = np.einsum("ijk,ij->ik", J, diff)  # (N, dim)
        grad = c1 * diff_norm ** (self.p - 2) * grad_  # (N, dim)

        if compute_hessian:
            c2 = self.p * (self.p - 2) / N
            H = np.array([self.hess(x) for x in X])  # (N, n_objective, dim, dim)
            idx = diff_norm != 0
            # some `diff` can be zero
            diff_norm_ = np.zeros(diff_norm.shape)
            diff_norm_[idx] = diff_norm[idx] ** (self.p - 4)
            diff_norm_ = diff_norm_[..., np.newaxis]
            # TODO: test this part for p != 2
            term = (
                c2 * np.tile(diff_norm_, (1, dim, dim)) * np.einsum("ij,ik->ijk", grad_, grad_)
                if self.p != 2
                else 0
            )
            hessian = term + c1 * (
                np.einsum("ijk,ijl->ikl", J, J) + np.einsum("ijkl,ij->ikl", H, diff)
            )  # (N, dim, dim)
            return grad, hessian
        else:
            return grad


class InvertedGenerationalDistance:
    def __init__(
        self,
        ref: Union[np.ndarray, ReferenceSet],
        func: Callable = None,
        jac: Callable = None,
        hess: Callable = None,
        p: float = 2,
        matching: bool = False,
    ):
        """Generational Distance

        Args:
            ref (np.ndarray): the reference set, of shape (n_point, n_objective)
            func (Callable): the objective function, which returns an array of shape (n_objective, )
            jac (Callable): the Jacobian function, which returns an array of shape (n_objective, dim)
            hess (Callable): the Hessian function, which returns an array of shape (n_objective, dim, dim)
            p (float, optional): parameter in the p-norm. Defaults to 2.
        """
        if isinstance(ref, np.ndarray):
            ref = ReferenceSet(ref)
        self.ref = ref
        self.p = p
        self.func = func
        self.jac = jac
        self.hess = hess
        self.M = self.ref.N
        self.matching = matching
        self.re_match = True

    def __str__(self) -> str:
        return "IGD (w/matching)" if self.matching else "IGD"

    def _compute_indices(self, Y: np.ndarray):
        """find for each reference point, the index of its closest point in the approximation set

        Args:
            Y (np.ndarray): the objective points of shape (N, n_objective).
        """
        N = len(Y)
        self.D = cdist(self.ref.reference_set, Y, metric="minkowski")
        # for each reference point, the index of its closest point in the approximation set `Y`
        self._indices = np.argmin(self.D, axis=1)
        # for each point `p`` in `Y`, the indices of points in the reference set
        # which have `p` as the closest point.
        self.indices = [np.nonzero(self._indices == i)[0] for i in range(N)]
        # for each point `p` in `Y`, the total number of points in the reference set
        # which have `p` as the closest point
        self.m = np.zeros((N, 1))
        idx, counts = np.unique(self._indices, return_counts=True)
        self.m[idx, 0] = counts

    def compute(self, X: np.ndarray = None, Y: np.ndarray = None) -> float:
        """compute the inverted generational distance value

        Args:
            X (np.ndarray, optional): the decision points of shape (N, dim). Defaults to None.
            Y (np.ndarray, optional): the objective points of shape (N, n_objective). Defaults to None.
                `X` and `Y` cannot be None at the same time

        Returns:
            float: the indicator value
        """
        if Y is None:
            assert X is not None
            assert self.func is not None
            Y = np.array([self.func(x) for x in X])
        if self.matching:
            self.ref.match(Y)
            return np.mean(np.sum((Y - self.ref.reference_set) ** 2, axis=1))
        else:
            self._compute_indices(Y)
            return np.mean(self.D[np.arange(self.M), self._indices] ** self.p) ** (1 / self.p)

    def compute_derivatives(
        self,
        X: np.ndarray,
        Y: np.ndarray = None,
        compute_hessian: bool = True,
        Jacobian: np.ndarray = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """compute the derivatives of the inverted generational distance^p

        Args:
            X (np.ndarray): the decision points of shape (N, dim).
            Y (np.ndarray, optional): the objective points of shape (N, n_objective). Defaults to None.
            compute_hessian (bool, optional): whether the Hessian is computed. Defaults to True.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                if `compute_hessian` = True, it returns (gradient, Hessian)
                otherwise, it returns (gradient, )
        """
        assert self.jac is not None
        assert self.hess is not None
        # TODO: implement p != 2
        # NOTE: for the matching method, `self.M = 1` since it is one-to-one matching
        c = 2 if self.matching else 2 / self.M
        dim = X.shape[1]
        if Y is None:
            Y = np.array([self.func(x) for x in X])
        # Jacobian of the objective function
        if Jacobian is None:
            J = np.array([self.jac(x) for x in X])  # (N, n_obj, dim)
        else:
            J = Jacobian
        if self.matching:
            self.ref.match(Y)
            diff = Y - self.ref.reference_set  # (N, n_obj)
        else:
            self._compute_indices(Y)
            Z = np.array([np.sum(self.ref.reference_set[idx], axis=0) for idx in self.indices])  # (N, n_obj)
            diff = self.m * Y - Z

        grad = c * np.einsum("ijk,ij->ik", J, diff)  # (N, dim)
        if compute_hessian:
            H = np.array([self.hess(x) for x in X])  # (N, n_obj, dim, dim)
            m = np.tile(self.m[..., np.newaxis], (1, dim, dim)) if not self.matching else 1
            hessian = c * (
                m * np.einsum("ijk,ijl->ikl", J, J) + np.einsum("ijkl,ij->ikl", H, diff)
            )  # (N, dim, dim)
            return grad, hessian
        else:
            return grad
