import mpmath as mp
import numpy as np

mp2np = lambda x: np.array(x.tolist(), dtype=float)
np2mp = lambda x: mp.matrix(x.tolist())


def flatten(X, order="C"):
    return mp.matrix([_ for _ in X]).T if order == "C" else mp.matrix([_ for _ in X.T]).T


def reshape(X: mp.matrix, R: int, C: int):
    assert R != -1 or C != -1
    N = X.rows * X.cols
    X_ = [_ for _ in X]
    if R == -1:
        R = int(N / C)
    if C == -1:
        C = int(N / R)
    return mp.matrix([X_[i * C : (i + 1) * C] for i in range(R)])


def r_(*args):
    out = []
    for x in args:
        if isinstance(x, mp.mpf):
            out.append([x])
        else:
            out += x.tolist()
    return mp.matrix(out)


def c_(*args):
    out = []
    for x in args:
        if isinstance(x, mp.mpf):
            out.append([x])
        else:
            out += x.T.tolist()
    return mp.matrix(out).T


def get_rows(X, idx):
    return r_(*[X[i, :] for i in idx])


def set_rows(X, idx, v):
    for i in idx:
        X[i, :] = v[i, :]


def block_diag(*args):
    N = len(args)
    R, C = args[0].rows, args[0].cols
    out = mp.zeros(N * R, N * C)
    for i, x in enumerate(args):
        out[i * R : (i + 1) * R, i * C : (i + 1) * C] = x
    return out
