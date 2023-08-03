import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hvd.interpolate import ReferenceSetInterpolation

data = pd.read_csv("./data/DTLZ7_NSGA-II.csv")
# take the Pareto approximation points from the last two iterations
X = data[(data.run == 0) & (data.iteration.between(499, 500))].iloc[:, -3:].values
ip = ReferenceSetInterpolation(n_objective=3)
res = ip.interpolate(X)

XX = np.concatenate(res, axis=0)

fig = plt.figure(figsize=plt.figaspect(1 / 3.0))
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.view_init(50, -25)
ax.plot(X[:, 0], X[:, 1], X[:, 2], "g.", ms=8, alpha=0.6)
ax.plot(XX[:, 0], XX[:, 1], XX[:, 2], "r+", ms=6, alpha=0.6)
plt.tight_layout()
plt.show()
