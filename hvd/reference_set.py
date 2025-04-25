from functools import cached_property
from typing import Dict, List, Optional, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, directed_hausdorff
from sklearn_extra.cluster import KMedoids

from .utils import compute_chim, plot_reference_set_matching

__authors__ = ["Hao Wang"]


class ReferenceSet:
    def __init__(
        self,
        ref: Union[Dict[int, np.ndarray], np.ndarray],
        eta: Union[Dict[int, np.ndarray], np.ndarray] = None,
        Y_idx: Optional[np.ndarray] = None,  # TODO: `Y_idx` should not be passed in here.
        plot: bool = False,
    ) -> None:
        """Clustered Reference Set

        Args:
            ref (Union[Dict[int, np.ndarray], np.ndarray]): the reference set
            eta (Union[Dict[int, np.ndarray], np.ndarray], optional): direction to shift each component of
                the reference set. Defaults to None.
            Y_idx (Optional[np.ndarray], optional): indices to indicate the cluster of approximation set.
                Defaults to None.
        """
        self._ref: Dict[int, np.ndarray] = {0: ref} if isinstance(ref, np.ndarray) else ref
        assert isinstance(self._ref, dict)
        self.n_components: int = len(self._ref)  # the number of connected components of the reference set
        self.n_obj: int = self._ref[0].shape[1]
        self.Y_idx: List[np.ndarray] = Y_idx  # list of indices of clusters of the approximation set
        self.re_match: bool = True
        self.eta: Dict[int, np.ndarray] = eta
        if self.eta is None:
            self._compute_shift_direction()
        self._medoids: Dict[int, np.ndarray] = {i: None for i in range(self.n_components)}
        self.plot = plot

    @property
    def N(self) -> int:
        return len(self.reference_set)

    @property
    def reference_set(self) -> np.ndarray:
        return (
            self._matched_medoids
            if hasattr(self, "_matched_medoids")
            else np.concatenate(list(self._ref.values()), axis=0)
        )

    def match(self, Y: np.ndarray) -> np.ndarray:
        if not self.re_match and hasattr(self, "_matched_medoids"):
            return

        Y = self._partition_Y(Y)
        N = sum([len(y) for y in Y])
        # match the components of the reference set and `Y`
        idx = self._match_components(Y)
        out = np.zeros((N, self.n_obj))
        self._medoids_idx = np.empty(N, dtype=object)
        for k, v in enumerate(idx):
            Y_ = Y[v]
            assert len(self._ref[k]) >= len(Y_)
            try:
                medoids = self._medoids[k]
                assert len(medoids) == len(Y_)
            except:
                medoids = self._cluster(component=self._ref[k], k=len(Y_))
            medoids = self._match(medoids, Y_)
            self._medoids[k] = medoids
            out[self.Y_idx[v]] = medoids
            for i, j in enumerate(self.Y_idx[v]):
                self._medoids_idx[j] = (k, i)
        self._matched_medoids = out

        if self.plot:
            plot_reference_set_matching(self._matched_medoids, Y)

    def shift(self, c: float = 0.05, indices: np.ndarray = None):
        """Shift the reference set. It operates in two cases:
        - shift the medoids if `self.match` has been called, representing a reference set matched to the
            approximation set.
        - shift the entire reference set if`self.match` is never called.

        Args:
            c (float, optional): the step-size of the shift. Defaults to 0.05.
            indices (np.ndarray, optional): the indices of the medoids to shift. Defaults to None.
        """
        if hasattr(self, "_matched_medoids"):
            if indices is None:  # shift all medoids if indices is None
                indices = np.arange(len(self._matched_medoids))
            for k in indices:
                v = c * self.eta[self._medoids_idx[k][0]]
                # TODO: also shift the reference set here
                self._matched_medoids[k] += v
                self.set_medoids(self._matched_medoids[k], k)
        else:  # if the `self.match` is not called yet
            for k, ref in self._ref.items():
                self._ref[k] = ref + c * self.eta[k]

    def set_medoids(self, medoid: np.ndarray, k: int):
        # update the medoids
        c, idx = self._medoids_idx[k]
        self._medoids[c][idx] = medoid

    def _partition_Y(self, Y: np.ndarray) -> List[np.ndarray]:
        """partition the approximation set `Y`. We consider the following scenarios:
        - If the reference set is not clustered, then `Y` should not be partitioned.
           In this case, `self.Y_idx` is ignored.
        - If the reference set is clustered (`self.n_components` indicates the number), then
            - we take the partitioning of `Y` indicated by `self.Y_idx` if it is not `None`;
            - otherwise, we assign each element of `Y` to the closest cluster of `self._ref`

        Args:
            Y (np.ndarray): the approximation set of shape (N, `self.dim`)

        Returns:
            List[np.ndarray]: a list of partitions of `Y`
        """
        if self.n_components == 1:
            self.Y_idx = [list(range(len(Y)))]
            Y_partitions = [Y]
        else:
            # if the clustering/partition of `Y` is not given, we partition `Y` in the way that
            # each point is assigned to the closest cluster of the reference set
            if self.Y_idx is None:
                D = np.array([cdist(self._ref[i], Y).min(axis=0) for i in range(self.n_components)])
                labels = np.argmin(D, axis=0)
                self.Y_idx = [np.nonzero(labels == i)[0] for i in range(self.n_components)]
            # parition `Y` according to `Y_idx`
            Y_partitions = [Y[idx] for idx in self.Y_idx]
        return Y_partitions

    def _match_components(self, Y: Dict[int, np.ndarray]) -> np.ndarray:
        n, m = self.n_components, len(Y)
        assert n >= m  # TODO: find a solution here
        if n == 1 and m == 1:
            idx = [0]
        else:
            # match the clusters of `X` and `Y`
            idx = np.zeros(m)
            cost = np.array([directed_hausdorff(self._ref[i], Y[j])[0] for i in range(n) for j in range(m)])
            cost = cost.reshape(n, m)
            idx = linear_sum_assignment(cost)[1]
        return idx

    def _cluster(self, component: np.ndarray, k: int) -> np.ndarray:
        """cluster the reference components with k-medoids method
        Args:
            component (np.ndarray): a component of the reference set
            K (int): the number of medoids

        Returns:
            np.ndarray: _description_
        """
        if len(component) == k:
            return component
        km = KMedoids(n_clusters=k, method="alternate", random_state=0, init="k-medoids++").fit(component)
        return component[km.medoid_indices_]

    def _match(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        cost = cdist(Y, X)
        idx = linear_sum_assignment(cost)[1]  # min-weight assignment in a bipartite graph
        return X[idx]

    def _compute_shift_direction(self):
        self.eta = {k: compute_chim(component) for k, component in self._ref.items()}
