import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from hvd.algorithm import HVN
from hvd.hypervolume_derivatives import get_non_dominated
from matplotlib import rcParams
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import qmc

plt.style.use("ggplot")
rcParams["font.size"] = 12
rcParams["xtick.direction"] = "out"
rcParams["ytick.direction"] = "out"
rcParams["text.usetex"] = True
rcParams["legend.numpoints"] = 1
rcParams["xtick.labelsize"] = 12
rcParams["ytick.labelsize"] = 12
rcParams["xtick.major.size"] = 7
rcParams["xtick.major.width"] = 1
rcParams["ytick.major.size"] = 7
rcParams["ytick.major.width"] = 1

G_norm = np.load("Eq1DTLZ1.npz")["G_norm"]
M, N = G_norm.shape

median = np.mean(G_norm, axis=0)
sd = np.std(G_norm, axis=0) / np.sqrt(M)

fig, ax = plt.subplots(1, 1)
x = list(range(1, N + 1))
ax.plot(x, median, "g-")
ax.fill_between(x, median - 1.96 * sd, median + 1.96 * sd, fc="g", interpolate=True, ec="none", alpha=0.4)
ax.set_ylabel(r"$||G(\mathbf{X})||$", color="g")
ax.set_title("Performance")
ax.set_xlabel("iteration")
ax.set_yscale("log")
ax.set_xticks(x)

plt.tight_layout()
plt.show()
plt.savefig(f"Eq1DTLZ1-avg.pdf", dpi=100)

df = pd.DataFrame(dict(iteration=range(1, N + 1), G_norm_mean=median, G_norm_sd=sd))
df.to_latex(buf=f"Eq1DTLZ1-avg.tex", index=False)
