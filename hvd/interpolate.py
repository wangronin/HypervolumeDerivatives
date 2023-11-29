import math
from itertools import combinations
from typing import List

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

from .utils import non_domin_sort

# TODO: this code is not tested yet


def get_vol(simplex: np.ndarray) -> float:
    """Compute the volume via the Cayley-Menger determinant
    <http://mathworld.wolfram.com/Cayley-MengerDeterminant.html>. One advantage is
    that it can compute the volume of the simplex independent of the dimension of the
    space in which it is embedded.

    Args:
        simplex (np.ndarray): _description_

    Returns:
        float: the volume of the simplex
    """
    # compute all edge lengths
    edges = np.subtract(simplex[:, None], simplex[None, :])
    ei_dot_ej = np.einsum("...k,...k->...", edges, edges)

    j = simplex.shape[0] - 1
    a = np.empty((j + 2, j + 2) + ei_dot_ej.shape[2:])
    a[1:, 1:] = ei_dot_ej
    a[0, 1:] = 1.0
    a[1:, 0] = 1.0
    a[0, 0] = 0.0

    a = np.moveaxis(a, (0, 1), (-2, -1))
    det = np.linalg.det(a)
    vol = np.sqrt((-1.0) ** (j + 1) / 2**j / math.factorial(j) ** 2 * det)
    return vol


class ReferenceSetInterpolation:
    def __init__(self, n_objective: int) -> None:
        self.n_objective = n_objective

    def interpolate(self, data: np.ndarray, N: int = 1600, return_clusters: bool = True) -> np.ndarray:
        # take the non-dominated subset of the input
        data_ = np.array(non_domin_sort(data, only_front_indices=False)[0])
        # remove the outliers first
        score = LocalOutlierFactor(n_neighbors=5).fit_predict(data_)
        data_ = data_[score != -1]
        # clustering
        clusters = self._cluster(data_)

        # TODO: improve code structure
        out = list()
        volume = 0
        ch = list()
        for cluster in clusters:
            # compute the convex hull of `data` first
            _ch = ConvexHull(cluster, qhull_options="Qs")
            ch.append(_ch)
            volume += _ch.volume

        for i, cluster in enumerate(clusters):
            vertices = cluster[ch[i].vertices]
            # Delaunay triangulation
            tri = Delaunay(vertices)
            out.append(
                np.vstack([self._interpolate_simplex(vertices[idx], volume, N) for idx in tri.simplices])
            )
        return out if return_clusters else np.concatenate(out, axis=0)

    def _interpolate_simplex(self, vertices: np.ndarray, total_volume: float, N: int) -> np.ndarray:
        """Sample u.a.r. from an N-dimensional simplex

        Args:
            vertices (np.ndarray): vertices of the simplex
            total_volume (float): total_volume of all simplices
            N (int): the number of points to sample

        Returns:
            np.ndarray: the uniform samples
        """
        m = len(vertices)
        try:
            N_ = int(np.ceil(N * get_vol(vertices) / total_volume))
            w = np.c_[np.zeros((N_, 1)), np.sort(np.random.rand(N_, m - 1), axis=1), np.ones((N_, 1))]
            w = w[:, 1:] - w[:, 0:-1]
            return w @ vertices
        except:
            return vertices

    def _cluster(self, data: np.ndarray) -> np.ndarray:
        # perform a grid search for the hyperparameters
        # from 1/10 to 1/3 of the mean pairwise distance
        # minimal samples from 5 to 15 for a core point
        eps = np.arange(0.1, 0.3, 0.025) * np.mean(pdist(data))
        min_samples = list(range(5, 15))
        labels = DBSCAN(eps=eps[0], min_samples=10).fit(data).labels_
        clusters = [data[labels == i] for i in np.unique(labels)]  # if i != -1]  # filter out outliers
        return clusters

    def _compute_weakest_link_cluster(self, clusters: List[np.ndarray]) -> float:
        max_intra_cluster = np.array([pdist(c).max() if len(c) > 1 else 0 for c in clusters])
        min_inter_cluster = np.min([cdist(*p).min() for p in combinations(clusters, 2)])
        intra_cluster_wlp = np.max(
            [
                self._weakest_link_points(pair[0], pair[1], max_intra_cluster, clusters)
                for c in clusters
                for pair in combinations(c, 2)
            ]
        )
        return intra_cluster_wlp / min_inter_cluster

    def _weakest_link_points(
        self, p: np.ndarray, q: np.ndarray, max_intra_cluster_distance: np.ndarray, clusters: List[np.ndarray]
    ) -> float:
        # TODO: this function is NOT the correct implementation of the original paper!!
        return np.min(
            [
                np.max([max_intra_cluster_distance[i], np.linalg.norm(p - c[0]), np.linalg.norm(q - c[-1])])
                for i, c in enumerate(clusters)
            ]
        )
