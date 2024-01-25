import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

gen = 300
pvalues = []
stats = []
stats_func = lambda x: [np.median(x), np.quantile(x, 0.9) - np.quantile(x, 0.1)]
MOEAs = [
    "NSGA-II",
    # "NSGA-III",
    # "SMS-EMOA",
]
problems = [
    "CF1",
    "CF2",
    "CF3",
    "CF4",
    "CF5",
    "CF6",
    "CF7",
    # "ZDT1",
    # "ZDT2",
    # "ZDT3",
    # "ZDT4",
    # "ZDT6",
    # "DTLZ1",
    # "DTLZ2",
    # "DTLZ3",
    # "DTLZ4",
    # "DTLZ5",
    # "DTLZ6",
    # "DTLZ7",
]

for moea in MOEAs:
    for problem in problems:
        data1 = pd.read_csv(f"./results/{problem}-DpN-{moea}-{gen}.csv")
        data2 = pd.read_csv(f"./results/{problem}-{moea}-{gen}.csv")
        x, y = np.maximum(data1.GD.values, data1.IGD.values), np.maximum(data2.GD.values, data2.IGD.values)
        # filtering out the outliers in DpN
        q = np.quantile(x, q=(0.25, 0.75))
        iqr = q[1] - q[0]
        x = x[(x > q[0] - 1.5 * iqr) & (x < q[1] + 1.5 * iqr)]
        pvalue = mannwhitneyu(x=x, y=y, alternative="two-sided").pvalue
        pvalues.append(pvalue)
        stats.append([stats_func(x), stats_func(y)])

reject, pvals_corrected, _, _ = multipletests(pvalues, alpha=0.05, method="fdr_bh")
win, tie, loose = 0, 0, 0
for k, (n, m) in enumerate(stats):
    x, y = n[0], m[0]
    if reject[k]:
        if x < y:
            win += 1
            s = "$\\uparrow$"
        else:
            loose += 1
            s = "$\\downarrow$"
    else:
        tie += 1
        s = "$\\leftrightarrow$"
    stats[k] = [f"{n[0]:.4f}({n[1]:.4e}){s}", f"{m[0]:.4f}({m[1]:.4e})"]

Method = np.repeat(MOEAs, len(problems)).reshape(-1, 1)
Problem = np.tile(problems, len(MOEAs)).reshape(-1, 1)
summary = ["+/$\\approx$/-", "", f"{win}/{tie}/{loose}", ""]
data = pd.DataFrame(
    np.vstack([np.hstack([Method, Problem, stats]), summary]),
    columns=["Method", "Problem", "Newton (iter = 5)", "MOEA (iter = 300 + 5 \\times (4 + 10n)"],
)
print(data)
caption = """Warmstarting the Newton method after 300 iterations of the MOEA, we show the IGD values 
(median and 10\\% - 90\\% quantile range) of the final Pareto fronts, averaged over
30 independent runs. The Newton method is executed for five iterations, and the corresponding MOEA terminates
with 4\,870 iterations on ZDTs and 3\,700 on DTLZs. Mann–Whitney U test (with 5\\% significance level) is 
employed to compare the performance of the Newton method and the MOEA, where Holm-Sidak method is used to 
adjust the $p$-value for multiple testing. 
"""
data.to_latex(f"Newton-{gen}.tex", index=False, caption=caption)
