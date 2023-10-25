from typing import Callable, Tuple, Union

import numpy as np
from scipy.linalg import block_diag, cho_solve, cholesky, solve
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import LocalOutlierFactor
from sklearn_extra.cluster import KMedoids

__authors__ = ["Hao Wang"]


def precondition_hessian(H: np.ndarray) -> np.ndarray:
    """Precondition the Hessian matrix to make sure it is positive definite

    Args:
        H (np.ndarray): the Hessian matrix

    Returns:
        np.ndarray: the lower triagular decomposition of the preconditioned Hessian
    """
    # pre-condition the Hessian
    try:
        L = cholesky(H, lower=True)
    except:
        beta = 1e-6
        v = np.min(np.diag(H))
        tau = 0 if v > 0 else -v + beta
        I = np.eye(H.shape[0])
        for _ in range(35):
            try:
                L = cholesky(H + tau * I, lower=True)
                break
            except:
                tau = max(2 * tau, beta)
        else:
            print("Pre-conditioning the HV Hessian failed")
    return L


def preprocess(X: np.ndarray) -> np.ndarray:
    """remove duplicated points and outliers"""
    N = len(X)
    idx = []
    # remove duplicated points
    for i in range(N):
        x = X[i]
        CON = np.all(
            np.isclose(
                np.asarray(X[np.arange(N) != i], dtype="float"),
                np.asarray(x, dtype="float"),
            ),
            axis=1,
        )
        if all(~CON):
            idx.append(i)

    X = X[idx]
    score = LocalOutlierFactor(n_neighbors=int(N / 10)).fit_predict(X)
    return X[score != -1]


class GenerationalDistance:
    def __init__(self, ref: np.ndarray, func: Callable, jac: Callable, hess: Callable, p: float = 2):
        """Generational Distance

        Args:
            ref (np.ndarray): the reference set, of shape (n_point, n_objective)
            func (Callable): the objective function, which returns an array of shape (n_objective, )
            jac (Callable): the Jacobian function, which returns an array of shape (n_objective, dim)
            hess (Callable): the Hessian function, which returns an array of shape (n_objective, dim, dim)
            p (float, optional): parameter in the p-norm. Defaults to 2.
        """
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
        self, X: np.ndarray, Y: np.ndarray = None, compute_hessian: bool = True
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
        J = np.array([self.jac(x) for x in X])  # (N, n_objective, dim)
        diff = Y - self.ref[self.indices]  # (N, n_objective)
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
        ref: np.ndarray,
        func: Callable = None,
        jac: Callable = None,
        hess: Callable = None,
        p: float = 2,
        recursive: bool = False,
        cluster_matching: bool = False,
    ):
        """Generational Distance

        Args:
            ref (np.ndarray): the reference set, of shape (n_point, n_objective)
            func (Callable): the objective function, which returns an array of shape (n_objective, )
            jac (Callable): the Jacobian function, which returns an array of shape (n_objective, dim)
            hess (Callable): the Hessian function, which returns an array of shape (n_objective, dim, dim)
            p (float, optional): parameter in the p-norm. Defaults to 2.
            recursive (bool, optional): if true, it computes non-zero derivatives/gradiants for all the
                the points in the approximation set by recursively removing the points which has nonzero
                gradient (the nearest point to some reference point). Defaults to False.
        """
        self.ref = preprocess(ref)
        self.ref = ref
        self.p = p
        self.func = func
        self.jac = jac
        self.hess = hess
        self.M = len(self.ref)
        self.recursive = recursive
        self.cluster_matching = cluster_matching

    def _compute_indices(self, Y: np.ndarray):
        """find for each reference point, the index of its closest point in the approximation set

        Args:
            Y (np.ndarray): the objective points of shape (N, n_objective).
        """
        N = len(Y)
        self.D = cdist(Y, self.ref, metric="minkowski", p=self.p)
        # for each reference point, the index of its closest point in the approximation set `Y`
        self._indices = np.argmin(self.D, axis=0)
        # for each point `p`` in `Y`, the indices of points in the reference set
        # which have `p` as the closest point.
        self.indices = [np.array([])] * N
        # for each point `p` in `Y`, the total number of points in the reference set
        # which have `p` as the closest point
        self.m = np.zeros((N, 1))
        D = self.D.copy()
        while True:
            _indices = np.argmin(D, axis=0)
            pos = np.nonzero(self.m == 0)[0]
            for p in pos:
                self.indices[p] = np.nonzero(_indices == p)[0]
                self.m[p] = len(self.indices[p])
            if len(pos) == 0 or not self.recursive:
                break
            D[_indices, np.arange(self.M)] = np.inf

    def _cluster_reference_set(self, N: int):
        km = KMedoids(n_clusters=N, random_state=0, method="pam", max_iter=600).fit(self.ref)
        self._idx = km.medoid_indices_
        self._medroids = self.ref[km.medoid_indices_]

    def _match(self, Y: np.ndarray):
        N = len(Y)
        # if the clustering hasn't been done, or the number of approximation points changes
        if not hasattr(self, "_medroids") or len(self._medroids) != N:
            self._cluster_reference_set(N)
        cost = cdist(Y, self._medroids, metric="minkowski", p=self.p)
        # min-weight assignment in a bipartite graph
        self._medoids_idx = linear_sum_assignment(cost)[1]

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
        if self.cluster_matching:
            self._match(Y)
            return np.mean(np.sum((Y - self._medroids[self._medoids_idx]) ** 2, axis=1))
        else:
            self._compute_indices(Y)
            return np.mean(self.D[self._indices, np.arange(self.M)] ** self.p) ** (1 / self.p)

    def compute_derivatives(
        self, X: np.ndarray, Y: np.ndarray = None, compute_hessian: bool = True
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
        # TODO: implement p != 2
        # c = 2 / self.M
        assert self.jac is not None
        assert self.hess is not None
        c = 2
        dim = X.shape[1]
        if Y is None:
            Y = np.array([self.func(x) for x in X])
        # Jacobian of the objective function
        J = np.array([self.jac(x) for x in X])  # (N, n_objective, dim)
        if self.cluster_matching:
            self._match(Y)
            diff = Y - self._medroids[self._medoids_idx]  # (N, n_objective)
        else:
            self._compute_indices(Y)
            centroid = np.array([np.sum(self.ref[idx], axis=0) for idx in self.indices])
            diff = self.m * Y - centroid  # (N, n_objective)

        grad = c * np.einsum("ijk,ij->ik", J, diff)  # (N, dim)
        if compute_hessian:
            H = np.array([self.hess(x) for x in X])  # (N, n_objective, dim, dim)
            m = np.tile(self.m[..., np.newaxis], (1, dim, dim)) if not self.cluster_matching else 1
            hessian = c * (
                m * np.einsum("ijk,ijl->ikl", J, J) + np.einsum("ijkl,ij->ikl", H, diff)
            )  # (N, dim, dim)
            return grad, hessian
        else:
            return grad
