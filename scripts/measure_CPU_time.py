import sys
import time

sys.path.insert(0, "./")

import numpy as np
import pandas as pd

from hvd.problems import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, PymooProblemWithAD

np.random.seed(42)

N = 1e5
res = []
problems = [ZDT1(), ZDT2(), ZDT3(), ZDT4(), ZDT6()]
for problem in problems:
    f = PymooProblemWithAD(problem)
    FE_time = []
    for i in range(int(N)):
        x = np.random.rand(11)
        t0 = time.process_time_ns()
        Y = f.objective(x)
        t1 = time.process_time_ns()
        FE_time.append((t1 - t0) / 1e9)
    res.append(["FE", type(problem).__name__, np.mean(FE_time), np.median(FE_time), np.std(FE_time)])

N = 1e5
for problem in problems:
    f = PymooProblemWithAD(problem)
    FE_time = []
    func = f.objective
    jac = f.objective_jacobian
    hessian = f.objective_hessian
    for i in range(int(N)):
        x = np.random.rand(11)
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
        FE_time.append((t1 - t0) / 1e9)
    res.append(["AD", type(problem).__name__, np.mean(FE_time), np.median(FE_time), np.std(FE_time)])

df = pd.DataFrame(res, columns=["type", "problem", "mean_CPU", "median_CPU", "std_CPU"])
df.to_csv("ZDT_CPU_time.csv", index=False)
print(df)
#  type   problem  mean_CPU  median_CPU   std_CPU
# 0   FE  Eq1DTLZ1  0.000056    0.000054  0.000010
# 1   FE  Eq1DTLZ2  0.000058    0.000056  0.000010
# 2   FE  Eq1DTLZ3  0.000063    0.000060  0.000011
# 3   AD  Eq1DTLZ1  0.065438    0.058103  0.016560
# 4   AD  Eq1DTLZ2  0.058690    0.056689  0.006520
# 5   AD  Eq1DTLZ3  0.063028    0.062523  0.002414
