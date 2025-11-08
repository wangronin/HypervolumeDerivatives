import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


def _matlab_round(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def ComponentDetection(Py, eps, minpts):

    idx = DBSCAN(eps=eps, min_samples=minpts).fit(Py)

    idx.labels_

    cluesters_deteccioncomponentes = np.unique(idx.labels_)

    cluesters_deteccion_outnoise = cluesters_deteccioncomponentes[cluesters_deteccioncomponentes != -1]

    num_cluesters = len(cluesters_deteccion_outnoise)

    cluesters_idx_array = []
    for k in cluesters_deteccion_outnoise:
        cluesters_idx = np.where(idx.labels_ == k)[0]
        cluesters_idx_array.append(cluesters_idx)

    cluesters_Py_array = []
    for j in cluesters_idx_array:
        cluesters_Py = Py[j]
        cluesters_Py_array.append(cluesters_Py)

    return cluesters_Py_array, num_cluesters


def interpolate_tst(X: np.ndarray, Nf: int) -> np.ndarray:

    X = np.asarray(X, float)

    n, p = X.shape

    P = X[np.argsort(X[:, 0])]

    keep = np.ones(len(P), dtype=bool)
    keep[1:] = np.any(np.diff(P, axis=0) != 0, axis=1)
    P = P[keep]
    n = len(P)
    if n < 2:
        return np.repeat(P[:1], Nf, axis=0)

    seg = P[1:] - P[:-1]
    seglen = np.linalg.norm(seg, axis=1)
    s = np.zeros(n)
    s[1:] = np.cumsum(seglen)
    L = s[-1]
    if L == 0:
        return np.repeat(P[:1], Nf, axis=0)

    t = np.linspace(0.0, L, Nf)
    I = np.empty((Nf, p))
    for j in range(p):
        I[:, j] = np.interp(t, s, P[:, j])

    I[0, :], I[-1, :] = P[0, :], P[-1, :]
    return I


# ----------------Main Code------------------------------
def RSG(Py: np.ndarray, Nf: int, eps: float = 0.08, minpts: int = 2) -> np.ndarray:
    num_obj = Py.shape[1]

    # --------------------------------------------------------#
    # --------------------2D FILLING--------------------------#
    # --------------------------------------------------------#
    if num_obj == 2:
        C, num_cluesters = ComponentDetection(Py=Py, eps=eps, minpts=minpts)

        I = []

        comp_length = np.zeros((1, num_cluesters), dtype=float)

        for comp in range(num_cluesters):
            X = C[comp]
            n = X.shape[0]
            Points_index = np.argsort(X[:, 0], kind="mergesort")
            Points = X[Points_index]
            distances = np.zeros((1, n - 1), dtype=float)

            for i in range(n - 1):
                distances[0, i] = np.linalg.norm(Points[i + 1, :] - Points[i, :])

            comp_length[0, comp] = distances.sum()

        total_length = comp_length.sum()

        for i in range(num_cluesters):

            Nf_interpolate_tst = int(_matlab_round(Nf * comp_length[0, i] / total_length))
            interpolate_tst_I = interpolate_tst(C[i], Nf_interpolate_tst)
            I.append(interpolate_tst_I)

        comp_idx_tot = []
        Fy_rows = []

        for i in range(len(I)):
            Ii = I[i]

            for j in range(Ii.shape[0]):
                comp_idx_tot.append(i + 1)
                Fy_rows.append(Ii[j, :])

        Fy = np.vstack(Fy_rows) if len(Fy_rows) > 0 else np.zeros((0, Py.shape[1]), dtype=float)

    return Fy
