import sys
import time

sys.path.insert(0, "./")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

from hvd.problems import (
    CF1,
    CF2,
    CF3,
    CF4,
    CF5,
    CF6,
    CF7,
    CF8,
    CF9,
    CF10,
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ4,
    DTLZ5,
    DTLZ6,
    DTLZ7,
    IDTLZ1,
    IDTLZ2,
    IDTLZ3,
    IDTLZ4,
    ZDT1,
    ZDT2,
    ZDT3,
    ZDT4,
    ZDT6,
    MOOAnalytical,
    PymooProblemWithAD,
)

sns.set_theme(rc={"figure.figsize": (12, 6)})
np.random.seed(42)

plt.style.use("ggplot")
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

N = 1e5
res = []
data_FE = []
set_ = 1
if set_ == 1:
    problems_name = ["ZDT1", "ZDT2", "ZDT3", "ZDT4"] + [f"DTLZ{k}" for k in range(1, 8)]
    dict_ = locals()
    problems = [dict_[k]() for k in problems_name]
else:
    problems = [
        CF1(),
        CF2(),
        CF3(),
        CF4(),
        CF5(),
        CF6(),
        CF7(),
        CF8(),
        CF9(),
        CF10(),
        IDTLZ1(),
        IDTLZ2(),
        IDTLZ3(),
        IDTLZ4(),
    ]
    problems_name = [type(p).__name__ for p in problems]

for problem in problems:
    f = problem if isinstance(problem, MOOAnalytical) else PymooProblemWithAD(problem)
    FE_time = []
    for i in range(int(N)):
        r = problem.xu - problem.xl
        x = r * np.random.rand(problem.n_var) + problem.xl
        t0 = time.process_time_ns()
        Y = f.objective(x)
        t1 = time.process_time_ns()
        if i > 0:  # the first iteration contains JIT computation time
            FE_time.append((t1 - t0) / 1000.0)

    data_FE += FE_time
    res.append(
        ["FE", type(problem).__name__, np.mean(FE_time), np.median(FE_time), np.quantile(FE_time, 0.9)]
    )
data_FE = np.vstack([np.repeat(problems_name, N - 1), data_FE]).T

N = 1e5
data_AD = []
for problem in problems:
    f = problem if isinstance(problem, MOOAnalytical) else PymooProblemWithAD(problem)
    FE_time = []
    func = f.objective
    jac = f.objective_jacobian
    hessian = f.objective_hessian
    for i in range(int(N)):
        r = problem.xu - problem.xl
        x = r * np.random.rand(problem.n_var) + problem.xl
        # record the CPU time of function evaluations
        t0 = time.process_time_ns()
        # Y = func(x)  # `(N, dim_m)`
        # H = h(x)
        # Jacobians
        # YdX = jac(x)  # `(N, dim_m, dim_d)`
        # Hessians
        YdX2 = hessian(x)  # `(N, dim_m, dim_d, dim_d)`
        # HdX = h_jac(x)
        # HdX2 = h_hessian(x)
        t1 = time.process_time_ns()
        if i > 0:
            FE_time.append((t1 - t0) / 1000.0)
    data_AD += FE_time
    res.append(
        ["AD", type(problem).__name__, np.mean(FE_time), np.median(FE_time), np.quantile(FE_time, 0.9)]
    )

data_AD = np.vstack([np.repeat(problems_name, N - 1), data_AD]).T
data = pd.DataFrame(
    np.c_[
        ["function evaluation"] * len(data_FE) + ["automatic differentiation"] * len(data_AD),
        np.vstack([data_FE, data_AD]),
    ],
    columns=["type", "problem", "time"],
)
data.time = pd.to_numeric(data.time)
ratios = []
for p in problems_name:
    data_ = data[(data.problem == p)].groupby("type")["time"].mean()
    ratios.append(data_["automatic differentiation"] / data_["function evaluation"])
print(rf"CPU time ratio (AD/FE): {np.mean(ratios)}")

df = pd.DataFrame(res, columns=["type", "problem", "mean_CPU", "median_CPU", "upper quantile"])
df.to_csv("CPU_time.csv", index=False)
print(df)

from matplotlib.ticker import LogLocator

data = data.astype({"time": "float64"})
ax = sns.violinplot(
    data=data, x="problem", y="time", hue="type", log_scale=True, split=True, gap=0.1, inner="quart"
)
ax.set_ylabel("CPU time (sec)")
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
ax.set_ylim(2, 20)
ax.get_legend().set_title("")
plt.tight_layout()
fig = ax.get_figure()
fig.savefig("CPU_time.pdf")
#  type   problem  mean_CPU  median_CPU   std_CPU
# 0   FE  Eq1DTLZ1  0.000056    0.000054  0.000010
# 1   FE  Eq1DTLZ2  0.000058    0.000056  0.000010
# 2   FE  Eq1DTLZ3  0.000063    0.000060  0.000011
# 3   AD  Eq1DTLZ1  0.065438    0.058103  0.016560
# 4   AD  Eq1DTLZ2  0.058690    0.056689  0.006520
# 5   AD  Eq1DTLZ3  0.063028    0.062523  0.002414
