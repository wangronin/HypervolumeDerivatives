import sys
from typing import Tuple

sys.path.insert(0, "./")

import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from base import ParetoApproximator
from matplotlib import rcParams

from hvd.hypervolume import hypervolume
from hvd.newton import HVN

plt.style.use("ggplot")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
rcParams["font.size"] = 15
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

max_iters = 10
path = "./data/HVSS/"


def HV_subset_selection() -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray, np.ndarray]
):
    t0 = time.time()
    Y0 = pd.read_csv(f"{path}/DTLZ2_Ay.csv", header=None, index_col=False).values
    X0 = Y0[:, 0:-1]
    centers = pd.read_csv(f"{path}/DTLZ2_box_centers_f1f2.csv", header=None, index_col=False).values
    radius = pd.read_csv(f"{path}/DTLZ2_box_radius_f1f2.csv", header=None, index_col=False).values
    func = ParetoApproximator(Y0, centers, radius)
    ref = 1.5 * np.max(Y0, axis=0)  # set the reference point based on the data
    N = len(Y0)
    func.plot_domain()
    plt.show()
    breakpoint()
    opt = HVN(
        n_var=1,
        n_obj=2,
        ref=ref,
        func=func.objective,
        jac=func.jacobian,
        hessian=func.hessian,
        g=func.ieq_constraint,
        g_jac=func.ieq_jacobian,
        g_hessian=func.ieq_hessian,
        N=N,
        X0=X0,
        xl=func.xl,
        xu=func.xu,
        max_iters=max_iters,
        verbose=False,
        preconditioning=False,
    )
    Y = opt.run()[1]
    best_so_far_HV = np.maximum.accumulate(np.array(opt.history_indicator_value))
    best_so_far_R_norm = np.minimum.accumulate(np.array(opt.history_R_norm))
    HV0, HV1 = best_so_far_HV[0], best_so_far_HV[-1]
    # select points from the archive
    # mat = scipy.io.loadmat(f"{path}/A_{problem_name}_run{run_id}.mat")
    # archive_Y = mat["Ay"]
    # archive_Y = pd.read_csv(f"{path}/Ay_{problem_name}.csv", index_col=None, header=None).values
    # D = scipy.spatial.distance.cdist(Y, archive_Y, metric="euclidean")
    # indices = np.argmin(D, axis=1)
    # Y_sel = archive_Y[indices]
    wall_clock_time = time.time() - t0
    # HV2 = hypervolume(Y_sel, ref)
    return Y, indices, ref, HV0, HV1, HV2, wall_clock_time, best_so_far_HV, best_so_far_R_norm


HV_subset_selection()
