from typing import Union

import numpy as np
from scipy.interpolate import BSpline

coeff = np.array(
    [
        [0.001252528, 0.964662523, 0.0],
        [0.01947433331299664, 0.6705892382800092, 0.0],
        [0.27016246911506425, 0.4305933960365816, 0.0],
        [0.4863707860259108, 0.2834106933629399, 0.0],
        [0.7430099474586837, 0.12671565799862497, 0.0],
        [1.0, 1.5e-11, 0.0],
    ]
)
k = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
bs = BSpline(t=k, c=coeff[:, 0:2], k=5, extrapolate=False)
bs_jac = bs.derivative()
bs_hess = bs_jac.derivative()


def objective(t: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(t, np.ndarray):
        t = t[0]  # ensure the output shape is (2,)
    return bs(t)


def jacobian(t: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(t, np.ndarray):
        t = t[0]  # ensure the output shape is (2,1)
    # whenever a point is on the boundary, we set the corresponding Jacobian to zero
    # to make the point stationary
    if t == 0 or t == 1:
        return np.zeros((2, 1))
    else:
        return bs_jac(t).reshape(-1, 1)


def hessian(t: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(t, np.ndarray):
        t = t[0]  # ensure the output shape is (2, 1, 1)
    # whenever a point is on the boundary, we set the corresponding Jacobian to zero
    # to make the point stationary
    if t == 0 or t == 1:
        return np.zeros((2, 1, 1))
    else:
        return bs_hess(t).reshape(2, 1, 1)


if 11 < 2:  # sanity check
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import rcParams

    rcParams["font.size"] = 17
    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"
    rcParams["text.usetex"] = True
    rcParams["legend.numpoints"] = 1
    rcParams["xtick.labelsize"] = 17
    rcParams["ytick.labelsize"] = 17
    rcParams["xtick.major.size"] = 7
    rcParams["xtick.major.width"] = 1
    rcParams["ytick.major.size"] = 7
    rcParams["ytick.major.width"] = 1
    plt.style.use("ggplot")

    data = pd.read_csv("./examples/subset_selection/ZDT1/points.csv", header=None).values
    t = np.linspace(0, 1, 30)
    xy = np.array([bs(_) for _ in t])
    tangent_xy = np.array([bs_jac(_) for _ in t])
    tangent_xy /= np.sqrt(np.sum(tangent_xy**2, axis=1)).reshape(-1, 1)
    normal_xy = np.array([bs_hess(_) for _ in t])
    normal_xy /= np.sqrt(np.sum(normal_xy**2, axis=1)).reshape(-1, 1)

    lines = []
    q = plt.quiver(
        xy[:, 0],
        xy[:, 1],
        tangent_xy[:, 0] / 3,
        tangent_xy[:, 1] / 3,
        scale_units="xy",
        angles="xy",
        scale=1,
        color="g",
        width=0.003,
        alpha=0.5,
        headlength=4.7,
        headwidth=2.7,
    )
    lines.append(q)
    q = plt.quiver(
        xy[:, 0],
        xy[:, 1],
        normal_xy[:, 0] / 5,
        normal_xy[:, 1] / 5,
        scale_units="xy",
        angles="xy",
        scale=1,
        color="r",
        width=0.003,
        alpha=0.5,
        headlength=4.7,
        headwidth=2.7,
    )
    lines.append(q)
    lines += plt.plot(xy[:, 0], xy[:, 1], "k-")
    lines += plt.plot(data[:, 0], data[:, 1], "r.")

    ax = plt.gca()
    ax.set_aspect("equal")
    plt.legend(lines, ["Jacobian/tangent vec", "Hessian/normal vec", "B-Spline", "data point"])
    plt.tight_layout()
    plt.savefig(f"ZDT1_B_Spline_check.pdf", dpi=1000)
