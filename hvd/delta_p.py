from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, directed_hausdorff, pdist
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


# TODO: think of a better abstraction here.
class ReferenceSet:
    def __init__(self, ref: np.ndarray, p: float = 2) -> None:
        self._ref = {0: ref} if isinstance(ref, np.ndarray) else ref
        assert isinstance(self._ref, dict)
        self.n_components = len(self._ref)
        self.p = p
        self.dim = self._ref[0].shape[1]
        self.N = len(self.reference_set)
        self._medoids = {i: None for i in range(self.n_components)}

    @property
    def reference_set(self) -> np.ndarray:
        return np.concatenate(list(self._ref.values()), axis=0)

    def match(self, Y: Union[dict, np.ndarray], Y_idx: Union[List, None] = None) -> np.ndarray:
        Y, Y_idx = self._check_Y(Y, Y_idx)
        N = sum([len(y) for y in Y])
        # match the components of the reference set and `Y`
        idx = self._match_components(Y)
        out = np.zeros((N, self.dim))
        self._medoids_idx = np.empty(N, dtype=object)
        for k in range(self.n_components):
            Y_ = Y[idx[k]]
            assert len(self._ref[k]) >= len(Y_)
            try:
                medoids = self._medoids[k]
                assert len(medoids) == len(Y_)
            except:
                medoids = self._cluster(X=self._ref[k], N=len(Y_))
            medoids = self._match(medoids, Y_)
            self._medoids[k] = medoids
            out[Y_idx[k]] = medoids
            for i, j in enumerate(Y_idx[k]):
                self._medoids_idx[j] = (k, i)
        return out

    def set_medoids(self, medroid: np.ndarray, k: int):
        # update the medoids
        c, idx = self._medoids_idx[k]
        self._medoids[c][idx] = medroid

    def _check_Y(self, Y: Union[dict, np.ndarray], Y_idx: Union[List, None]) -> Tuple[np.ndarray, np.ndarray]:
        Y = [Y] if isinstance(Y, np.ndarray) else Y
        Y_idx = [list(range(len(Y[0])))] if Y_idx is None else Y_idx
        return Y, Y_idx

    def _match_components(self, Y: Dict[int, np.ndarray]) -> np.ndarray:
        n, m = self.n_components, len(Y)
        assert n >= m  # TODO: find a solution here
        if n == 1 and m == 1:
            idx = [0]
        else:
            # match the clusters of `X` and `Y`
            iu = np.triu_indices(n=n, m=m)
            cost = np.zeros((n, m))
            cost[iu] = [directed_hausdorff(self._ref[i], Y[j])[0] for (i, j) in zip(*iu)]
            cost = cost + cost.T
            idx = linear_sum_assignment(cost)[1]
        return idx

    def _cluster(self, X: np.ndarray, N: int) -> np.ndarray:
        km = KMedoids(n_clusters=N, random_state=0, init="k-medoids++").fit(X)
        return X[km.medoid_indices_]

    def _match(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        cost = cdist(Y, X, metric="minkowski", p=self.p)
        idx = linear_sum_assignment(cost)[1]  # min-weight assignment in a bipartite graph
        return X[idx]


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
        self.ref = ReferenceSet(ref, p)
        self.p = p
        self.func = func
        self.jac = jac
        self.hess = hess
        self.M = self.ref.N
        self.recursive = recursive
        self.cluster_matching = cluster_matching
        self.re_match = True

    def _compute_indices(self, Y: np.ndarray):
        """find for each reference point, the index of its closest point in the approximation set

        Args:
            Y (np.ndarray): the objective points of shape (N, n_objective).
        """
        N = len(Y)
        self.D = cdist(Y, self.ref.reference_set, metric="minkowski", p=self.p)
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

    def _match(self, Y: np.ndarray, Y_label: np.ndarray = None):
        if not self.re_match and hasattr(self, "_medoids"):
            return
        Y_ = Y
        Y_idx = None
        # if the reference set is clustered, then also try to cluster the approximation set
        if Y_label is not None:
            n_cluster = len(np.unique(Y_label))
            Y_idx = [np.nonzero(Y_label == i)[0] for i in range(n_cluster)]
            Y_ = [Y[idx] for idx in Y_idx]
        self._medoids = self.ref.match(Y_, Y_idx)

    def shift_medoids(self, v: np.ndarray, k: int):
        self._medoids[k] += v
        self.ref.set_medoids(self._medoids[k], k)

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
            return np.mean(np.sum((Y - self._medoids) ** 2, axis=1))
        else:
            self._compute_indices(Y)
            return np.mean(self.D[self._indices, np.arange(self.M)] ** self.p) ** (1 / self.p)

    def compute_derivatives(
        self,
        X: np.ndarray,
        Y: np.ndarray = None,
        Y_label: np.ndarray = None,
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
        c = 2 if self.cluster_matching else 2 / self.M
        dim = X.shape[1]
        if Y is None:
            Y = np.array([self.func(x) for x in X])
        # Jacobian of the objective function
        if Jacobian is None:
            J = np.array([self.jac(x) for x in X])  # (N, n_objective, dim)
        else:
            J = Jacobian
        if self.cluster_matching:
            self._match(Y, Y_label)
            diff = Y - self._medoids  # (N, n_objective)
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
