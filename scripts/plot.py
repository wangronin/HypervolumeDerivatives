import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

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

data = pd.read_csv("./result.csv", header=0, index_col=0)
data.columns = ["Method", "Problem", "DpN", "MOEA", "test"]
data["impr"] = np.clip((data["MOEA"] - data["DpN"]) / data["MOEA"], -2, np.inf)
df = data.pivot(index="Problem", columns="Method", values="impr")
df_color = data.pivot(index="Problem", columns="Method", values="test")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
plt.subplots_adjust(bottom=0.1, top=0.9)
index = df.index
bar_colors = ["tab:red"] * len(df["MOEAD"])

colors = {
    "uparrow": "tab:red",
    "downarrow": "tab:blue",
    "leftrightarrow": "tab:gray",
}
bar_colors = ["tab:red"] * 10 + [colors[_] for _ in df_color["MOEAD"].iloc[10:]]
ax1.bar(index, df["MOEAD"], color=bar_colors)
ax1.set_title("MOEA/D")
ax2.bar(index, df["NSGA-II"], color=[colors[_] for _ in df_color["NSGA-II"]])
ax2.set_title("NSGA-II")
ax3.bar(index, df["NSGA-III"], color=[colors[_] for _ in df_color["NSGA-III"]])
ax3.set_title("NSGA-III")
ax4.bar(index, df["SMS-EMOA"], color=[colors[_] for _ in df_color["SMS-EMOA"]])
ax4.set_title("SMS-EMOA")
fig.supylabel(r"Relative Improvement: $(\Delta_2$(MOEA) $-$ $\Delta_2$(hybrid))$/\Delta_2$(MOEA)")
plt.xticks(rotation=90)
fig.tight_layout()
plt.savefig(f"Newton-300.pdf", dpi=1000)

two_dim = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"] + [f"CF{i}" for i in range(1, 8)]
three_dim = [f"DTLZ{i}" for i in range(1, 8)] + [f"IDTLZ{i}" for i in range(1, 5)] + ["CF8", "CF9"]
four_dim = ["CONV4-2F"]
v = pd.DataFrame(
    {
        "bi-objective": data[data["Problem"].isin(two_dim)].groupby("Method")["impr"].mean(),
        "three-objective": data[data["Problem"].isin(three_dim)].groupby("Method")["impr"].mean(),
        # "four-objective": df[df["Problem"].isin(four_dim)].groupby("Method")["impr"].mean(),
    }
)
v.plot.bar(rot=0)

f = plt.gcf()
f.set_size_inches(10, 8)
plt.legend(loc=1, prop={"size": 20})
plt.ylabel("Average Relative Improvement over Problems")
plt.xlabel("")
f.tight_layout()
plt.savefig(f"relative_improvement_per_dim.pdf", dpi=1000)
