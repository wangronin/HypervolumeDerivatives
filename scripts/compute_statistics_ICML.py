import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

gen = 300  # the number of MOEA iterations
pvalues12 = []
pvalues13 = []
pvalues14 = []
stats = []
index = []
stats_func = lambda x: [np.median(x), np.quantile(x, 0.9) - np.quantile(x, 0.1)]
# MOEA algorithms to compare to
MOEAs = [
    "NSGA-II",
    "NSGA-III",
    "MOEAD",
]
method1 = "MMD"
method2 = "DpN"
method3 = "Lara"
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
            data1 = pd.read_csv(f"./results/ICML/{problem}-{method1}-{moea}-{gen}.csv")
            data2 = pd.read_csv(f"./results/ICML/{problem}-{method2}-{moea}-{gen}.csv")
            data3 = pd.read_csv(f"./results/ICML/{problem}-{method3}-{moea}-{gen}.csv")
            data4 = pd.read_csv(f"./results/ICML/{problem}-{moea}-{gen}.csv")
        except:
            continue
        x, y, z, c = (
            np.maximum(data1.GD.values, data1.IGD.values),
            np.maximum(data2.GD.values, data2.IGD.values),
            np.maximum(data3.GD.values, data3.IGD.values),
            np.maximum(data4.GD.values, data4.IGD.values),
        )
        pvalue = mannwhitneyu(x=x, y=y, alternative="two-sided").pvalue
        pvalues12.append(pvalue)
        pvalue = mannwhitneyu(x=x, y=z, alternative="two-sided").pvalue
        pvalues13.append(pvalue)
        pvalue = mannwhitneyu(x=x, y=c, alternative="two-sided").pvalue
        pvalues14.append(pvalue)
        stats.append([stats_func(x), stats_func(y), stats_func(z), stats_func(c)])
        index.append([moea, problem])

reject1, pvals_corrected, _, _ = multipletests(pvalues12, alpha=0.05, method="fdr_bh")
reject2, pvals_corrected, _, _ = multipletests(pvalues13, alpha=0.05, method="fdr_bh")
reject3, pvals_corrected, _, _ = multipletests(pvalues13, alpha=0.05, method="fdr_bh")
win1, tie1, loose1 = 0, 0, 0
win2, tie2, loose2 = 0, 0, 0
win3, tie3, loose3 = 0, 0, 0
for k, (n, m, q, p) in enumerate(stats):
    x, y, z, c = n[0], m[0], q[0], p[0]
    if reject1[k]:
        if x > y:
            win1 += 1
            s1 = "$\\uparrow$"
        else:
            loose1 += 1
            s1 = "$\\downarrow$"
    else:
        tie1 += 1
        s1 = "$\\leftrightarrow$"
    if reject2[k]:
        if x > z:
            win2 += 1
            s2 = "$\\uparrow$"
        else:
            loose2 += 1
            s2 = "$\\downarrow$"
    else:
        tie2 += 1
        s2 = "$\\leftrightarrow$"
    if reject3[k]:
        if x > c:
            win3 += 1
            s3 = "$\\uparrow$"
        else:
            loose3 += 1
            s3 = "$\\downarrow$"
    else:
        tie3 += 1
        s3 = "$\\leftrightarrow$"
    stats[k] = [
        f"{n[0]:.4f}({n[1]:.2e})",
        f"{m[0]:.4f}({m[1]:.2e}){s1}",
        f"{q[0]:.4f}({q[1]:.2e}){s2}",
        f"{p[0]:.4f}({p[1]:.2e}){s3}",
    ]

summary = [
    "+/$\\approx$/-",
    "",
    "",
    f"{win1}/{tie1}/{loose1}",
    f"{win2}/{tie2}/{loose2}",
    f"{win3}/{tie3}/{loose3}",
]
data = pd.DataFrame(
    np.vstack([np.hstack([index, stats]), summary]),
    columns=["Baseline", "Problem", "MMDN + MOEA", "DpN + MOEA", "Gradient-based", "MOEA alone"],
)
pd.set_option("display.max_rows", None)
print(data)
with open(f"{method1}-{method2}-{gen}.txt", "w") as file:
    print(data, file=file)

caption = f"""Warmstarting the {method1} and {method2} methods with the final population of MOEA 
executed for 300 iterations. Both methods are executed for five iterations, which corresponds to 
ca. 4\,870 iterations on ZDTs and 3\,700 on DTLZs for the MOEA. We show the $\Delta_p$ values 
(median and 10\\% - 90\\% quantile range) of the final Pareto fronts approximation, averaged over 
30 independent runs. Mann–Whitney U test (with 5\\% significance level) is employed to compare the 
performance of the Newton method and the MOEA, where Holm-Sidak method is used to adjust the 
$p$-value for multiple testing. 
"""
data.to_latex(f"{method1}-{method2}-{method3}-{gen}.tex", index=False, caption=caption)
