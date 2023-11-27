from typing import Callable, Dict, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, directed_hausdorff, pdist
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids

__authors__ = ["Hao Wang"]


class GenerationalDistance:
    def __init__(
        self,
        ref: np.ndarray,
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
        self.ref = np.concatenate([v for v in ref.values()], axis=0) if isinstance(ref, dict) else ref
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


class ClusteredMetroids:
    def __init__(self, n_components: int, p: float = 2, recluster: bool = False) -> None:
        self._medroids = {i: None for i in range(n_components)}
        self.recluster = recluster
        self.p = p

    def compute(
        self, X: Union[dict, np.ndarray], Y: Union[dict, np.ndarray], Y_idx: np.ndarray = None
    ) -> np.ndarray:
        # if we have single component
        X = [X] if isinstance(X, np.ndarray) else X
        Y = [Y] if isinstance(Y, np.ndarray) else Y
        n, m = len(X), len(Y)
        N, dim = np.sum([len(y) for y in Y]), X[0].shape[1]
        assert n >= m  # TODO: find a solution here
        if n == 1 and m == 1:
            idx = [0]
        else:
            # match the clusters of `X` and `Y`
            iu = np.triu_indices(n=n, m=m)
            cost = np.zeros((n, m))
            cost[iu] = [directed_hausdorff(X[i], Y[j])[0] for (i, j) in zip(*iu)]
            cost = cost + cost.T
            idx = linear_sum_assignment(cost)[1]
        # compute the medroids for each cluster of `X`
        self.Y_idx = [list(range(len(Y[0])))] if Y_idx is None else Y_idx
        out = np.zeros((N, dim))
        self._medroids_idx = np.empty(N, dtype=object)
        for k in range(n):
            Y_ = Y[idx[k]]
            assert len(X[k]) >= len(Y_)
            try:
                medroids = self._medroids[k]
                assert len(medroids) == len(Y_)
            except:
                medroids = self._cluster(X=X[k], N=len(Y_))
            medroids = self._match(medroids, Y_)
            self._medroids[k] = medroids
            out[self.Y_idx[k]] = medroids
            for i, j in enumerate(self.Y_idx[k]):
                self._medroids_idx[j] = (k, i)

        # number of medroids per each cluster
        self._n_medroids = [len(m) for m in self._medroids.values()]
        return out

    def _cluster(self, X: np.ndarray, N: int) -> np.ndarray:
        km = KMedoids(n_clusters=N, random_state=0, method="pam").fit(X)
        return X[km.medoid_indices_]

    def _match(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        cost = cdist(Y, X, metric="minkowski", p=self.p)
        idx = linear_sum_assignment(cost)[1]  # min-weight assignment in a bipartite graph
        return X[idx]

    def set_medroids(self, medroid: np.ndarray, k: int):
        # update the medroids
        c, idx = self._medroids_idx[k]
        self._medroids[c][idx] = medroid


class InvertedGenerationalDistance:
    def __init__(
        self,
        ref: np.ndarray,
        func: Callable = None,
        jac: Callable = None,
        hess: Callable = None,
        p: float = 2,
        recursive: bool = False,  # TODO: remove the `recursive` option
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
        # self.ref = preprocess_reference_set(ref)
        self.ref = ref
        self.p = p
        self.func = func
        self.jac = jac
        self.hess = hess
        self.M = len(self.ref)
        self.recursive = recursive
        self.cluster_matching = cluster_matching
        if self.cluster_matching:
            n_components = len(self.ref) if isinstance(self.ref, dict) else 1
            self._clustered_metroids = ClusteredMetroids(n_components, self.p)

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

    def set_medroids(self, v, k):
        self._medroids[k] = v
        self._clustered_metroids.set_medroids(v, k)

    def _match(self, Y: np.ndarray, Y_label: None):
        Y_ = Y
        Y_idx = None
        # if the reference set is clustered, then also try to cluster the approximation set
        if isinstance(self.ref, dict):
            # eps = 0.15 * np.min(pdist(Y))
            # label1 = DBSCAN(eps=0.5, min_samples=2).fit_predict(Y)
            n_cluster = len(np.unique(Y_label))
            Y_idx = [np.nonzero(Y_label == i)[0] for i in range(n_cluster)]
            Y_ = [Y[idx] for idx in Y_idx]
        self._medroids = self._clustered_metroids.compute(self.ref, Y_, Y_idx)

    def compute(self, X: np.ndarray = None, Y: np.ndarray = None, Y_label: np.ndarray = None) -> float:
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
            self._match(Y, Y_label)
            return np.mean(np.sum((Y - self._medroids) ** 2, axis=1))
        else:
            self._compute_indices(Y)
            return np.mean(self.D[self._indices, np.arange(self.M)] ** self.p) ** (1 / self.p)

    def compute_derivatives(
        self, X: np.ndarray, Y: np.ndarray = None, Y_label: np.ndarray = None, compute_hessian: bool = True
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
        c = 2 if self.cluster_matching else 2 / self.M
        dim = X.shape[1]
        if Y is None:
            Y = np.array([self.func(x) for x in X])
        # Jacobian of the objective function
        J = np.array([self.jac(x) for x in X])  # (N, n_objective, dim)
        if self.cluster_matching:
            self._match(Y, Y_label)
            diff = Y - self._medroids  # (N, n_objective)
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
