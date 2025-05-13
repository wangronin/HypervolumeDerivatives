import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

gen = 300  # the number of MOEA iterations
pvalues = []
stats = []
index = []
stats_func = lambda x: [np.median(x), np.quantile(x, 0.9) - np.quantile(x, 0.1)]
# MOEA algorithms to compare to
MOEAs = [
    "NSGA-II",
    "NSGA-III",
    "MOEAD",
]
method = "MMD"
# test problem
problems = [
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "ZDT4",
    "DTLZ1",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ6",
    "DTLZ7",
]

for moea in MOEAs:
    for problem in problems:
        try:
            data1 = pd.read_csv(f"./results/{problem}-{method}-{moea}-{gen}.csv")
            data2 = pd.read_csv(f"./results/{problem}-{moea}-{gen}.csv")
        except:
            continue
        x, y = np.maximum(data1.GD.values, data1.IGD.values), np.maximum(data2.GD.values, data2.IGD.values)
        pvalue = mannwhitneyu(x=x, y=y, alternative="two-sided").pvalue
        pvalues.append(pvalue)
        stats.append([stats_func(x), stats_func(y)])
        index.append([moea, problem])

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
    stats[k] = [f"{n[0]:.4f}({n[1]:.2e}){s}", f"{m[0]:.4f}({m[1]:.2e})"]

summary = ["+/$\\approx$/-", "", f"{win}/{tie}/{loose}", ""]
data = pd.DataFrame(
    np.vstack([np.hstack([index, stats]), summary]),
    columns=["Method", "Problem", "Newton (iter = 5)", "MOEA (iter = 300 + 5 \\times (4 + 10n)"],
)
pd.set_option("display.max_rows", None)
print(data)
with open(f"{method}-{gen}.txt", "w") as file:
    print(data, file=file)

caption = f"""Warmstarting the {method} method with the final population of MOEA executed for 300 iterations. 
The {method} is executed for five iterations, which corresponds to ca. 4\,870 iterations on ZDTs and 3\,700 on 
DTLZs for the MOEA. We show the $\Delta_p$ values (median and 10\\% - 90\\% quantile range) of the final 
Pareto fronts approximation, averaged over 30 independent runs. Mann–Whitney U test (with 5\\% significance 
level) is employed to compare the performance of the Newton method and the MOEA, where Holm-Sidak method is 
used to adjust the $p$-value for multiple testing. 
"""
data.to_latex(f"{method}-{gen}.tex", index=False, caption=caption)
