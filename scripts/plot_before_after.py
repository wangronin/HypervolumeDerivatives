import sys

sys.path.insert(0, "./")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from hvd.problems import CONV3, DTLZ1, DTLZ7
from hvd.problems.base import PymooProblemWithAD

plt.style.use("ggplot")
rcParams["font.size"] = 17
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["legend.fontsize"] = 10
rcParams["xtick.labelsize"] = 17
rcParams["ytick.labelsize"] = 17
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

np.random.seed(66)

f = DTLZ1()
# pareto_front = f.get_pareto_front(5000)
problem = PymooProblemWithAD(f)
pareto_front = problem.get_pareto_front()

df = pd.read_csv("DTLZ1_example.csv")

X0 = df[df.iteration == 0].values[:, 2:]
X1 = df[df.iteration == 10].values[:, 2:]

fig = plt.figure(figsize=plt.figaspect(1 / 2.0))
plt.subplots_adjust(bottom=0.05, top=0.95, right=0.93, left=0.05)
ax0 = fig.add_subplot(1, 2, 1, projection="3d")
ax0.set_box_aspect((1, 1, 1))
ax0.view_init(70, -20)

ax0.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "g.", mec="none", ms=5, alpha=0.8)
ax0.plot(X0[:, 0], X0[:, 1], X0[:, 2], "k.", mec="none", ms=8, alpha=0.5)

ax1 = fig.add_subplot(1, 2, 2, projection="3d")
ax1.set_box_aspect((1, 1, 1))
ax1.view_init(70, -20)

ax1.plot(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], "g.", mec="none", ms=5, alpha=0.8)
ax1.plot(X1[:, 0], X1[:, 1], X1[:, 2], "r.", mec="none", ms=8, alpha=0.5)
plt.show()
