import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

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

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True, figsize=(13, 5))

G_norm = np.load("Eq1DTLZ1.npz")["G_norm"]
M, N = G_norm.shape
median = np.mean(G_norm, axis=0)
sd = np.std(G_norm, axis=0) / np.sqrt(M)
x = list(range(1, N + 1))
df1 = pd.DataFrame(dict(G_norm_mean=median, G_norm_sd=sd))

ax0.plot(x, median, "g-")
ax0.fill_between(x, median - 1.96 * sd, median + 1.96 * sd, fc="g", interpolate=True, ec="none", alpha=0.4)
ax0.set_ylabel(r"$||G(\mathbf{X})||$", color="g")
ax0.set_xlabel("iteration")
ax0.set_yscale("log")
ax0.set_title("Eq1DTLZ1")

G_norm = np.load("Eq1DTLZ2.npz")["G_norm"]
M, N = G_norm.shape
median = np.mean(G_norm, axis=0)
sd = np.std(G_norm, axis=0) / np.sqrt(M)
x = list(range(1, N + 1))
df2 = pd.DataFrame(dict(G_norm_mean=median, G_norm_sd=sd))

ax1.plot(x, median, "g-")
ax1.fill_between(x, median - 1.96 * sd, median + 1.96 * sd, fc="g", interpolate=True, ec="none", alpha=0.4)
# ax0.set_ylabel(r"$||G(\mathbf{X})||$", color="g")
ax1.set_xlabel("iteration")
ax1.set_yscale("log")
ax1.set_title("Eq1DTLZ2")

G_norm = np.load("Eq1DTLZ3.npz")["G_norm"]
M, N = G_norm.shape
median = np.mean(G_norm, axis=0)
sd = np.std(G_norm, axis=0) / np.sqrt(M)
x = list(range(1, N + 1))
df3 = pd.DataFrame(dict(G_norm_mean=median, G_norm_sd=sd))

ax2.plot(x, median, "g-")
ax2.fill_between(x, median - 1.96 * sd, median + 1.96 * sd, fc="g", interpolate=True, ec="none", alpha=0.4)
# ax0.set_ylabel(r"$||G(\mathbf{X})||$", color="g")
ax2.set_xlabel("iteration")
ax2.set_yscale("log")
ax2.set_title("Eq1DTLZ3")

plt.tight_layout()
plt.savefig(f"Eq1DTLZ-avg.pdf", dpi=100)

columns = [(a, b) for a in ["Eq1DTLZ1", "Eq1DTLZ2", "Eq1DTLZ3"] for b in ["mean", "sd"]]
df = pd.concat([df1, df2, df3], axis=1)
df.columns = pd.MultiIndex.from_tuples(columns)
df.insert(0, "iteration", list(range(1, N + 1)))
df.to_latex(buf=f"Eq1DTLZ-avg.tex", index=False)
