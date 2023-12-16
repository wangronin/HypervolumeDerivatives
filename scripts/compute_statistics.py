import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

gen = 110
emoa = "NSGA-II"
# out = []
# dfs = []
pvalues = []
stats = []
func = lambda x: f"{np.median(x):.4e} +/- {np.std(x) / np.sqrt(len(x)):.4e}"

# problems = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
problems = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"]
for problem in problems:
    data1 = pd.read_csv(f"./results/{problem}-DpN-{emoa}-{gen}.csv")
    data2 = pd.read_csv(f"./results/{problem}-{emoa}-{gen}.csv")
    x, y = data1.IGD.values, data2.IGD.values
    # df1 = data1.loc[:, ["IGD"]]
    # df1.insert(0, "dataset_name", problem)
    # df1.insert(0, "classifier_name", "DpN")
    # df2 = data2.loc[:, ["IGD"]]
    # df2.insert(0, "dataset_name", problem)
    # df2.insert(0, "classifier_name", emoa)
    # dfs.append(pd.concat([df1, df2], axis=0))

    pvalue = mannwhitneyu(x=x, y=y, alternative="less").pvalue
    pvalues.append(pvalue)
    stats.append([func(x), func(y)])
    # df1 = data1.apply(func, axis=0)
    # df2 = data2.apply(lambda x: f"{np.median(x):.4e} +/- {np.std(x) / np.sqrt(len(x)):.4e}", axis=0)

# df = pd.concat(dfs, axis=0)
# df.rename(columns={"IGD": "accuracy"}, inplace=True)
# df.to_csv("example.csv", index=False)
reject, pvals_corrected, _, _ = multipletests(pvalues, alpha=0.05)
res = np.c_[np.vstack([np.array(stats).T, [f"{p:.4e}" for p in pvals_corrected]]), ["", "", sum(reject)]]
df = pd.DataFrame(
    res,
    columns=problems + ["Total"],
    index=["Newton", emoa, "p-value"],
)
df.to_csv(f"Newton-{emoa}-ZDT-{gen}.csv")
df.to_latex(f"Newton-{emoa}-ZDT-{gen}.tex")
