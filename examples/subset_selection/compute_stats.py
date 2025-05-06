import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

path = "./spline"
pvalues = []
stats = []
index = []
stats_func = lambda x: [np.median(x), np.quantile(x, 0.9) - np.quantile(x, 0.1)]
problems = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "DTLZ1", "DTLZ2", "DENT", "CONV2", "2on1"]
# metric = "wall clock time"
metric = "hypervolume"
if metric == "wall clock time":
    type_ = "t"
else:
    type_ = "HV"

stats_HVNS = []
stats_GAHSS = []
stats_LGISS = []
pvalues_GAHSS = []
pvalues_LGISS = []
for k, problem in enumerate(problems):
    data_HVNS = pd.read_csv(f"{path}/results/{type_}_HVN.csv", index_col=None, header=None).values[k]
    data_GAHSS = pd.read_csv(f"{path}/results/{type_}_GAHSS.csv", index_col=None, header=None).values[k]
    data_LGISS = pd.read_csv(f"{path}/results/{type_}_LGISS.csv", index_col=None, header=None).values[k]

    p1 = mannwhitneyu(x=data_HVNS, y=data_GAHSS, alternative="greater").pvalue
    p2 = mannwhitneyu(x=data_HVNS, y=data_LGISS, alternative="greater").pvalue
    pvalues_GAHSS.append(p1)
    pvalues_LGISS.append(p2)
    stats_HVNS.append(stats_func(data_HVNS))
    stats_GAHSS.append(stats_func(data_GAHSS))
    stats_LGISS.append(stats_func(data_LGISS))

for k, (n, m) in enumerate(stats_HVNS):
    stats_HVNS[k] = f"{n:.4f}({m:.4e})"

reject, pvals_corrected, _, _ = multipletests(pvalues_GAHSS, alpha=0.05, method="fdr_bh")
win1, tie1, loose1 = 0, 0, 0
for k, (n, m) in enumerate(stats_GAHSS):
    x, y = n, m
    if reject[k]:
        if x > y:
            win1 += 1
            s = "$\\uparrow$"
        else:
            loose1 += 1
            s = "$\\downarrow$"
    else:
        tie1 += 1
        s = "$\\leftrightarrow$"
    stats_GAHSS[k] = [f"{x:.4f}({y:.4e}){s}"]

reject, pvals_corrected, _, _ = multipletests(pvalues_LGISS, alpha=0.05, method="fdr_bh")
win2, tie2, loose2 = 0, 0, 0
for k, (n, m) in enumerate(stats_LGISS):
    x, y = n, m
    if reject[k]:
        if x > y:
            win2 += 1
            s = "$\\uparrow$"
        else:
            loose2 += 1
            s = "$\\downarrow$"
    else:
        tie2 += 1
        s = "$\\leftrightarrow$"
    stats_LGISS[k] = [f"{x:.4f}({y:.4e}){s}"]

summary = ["+/$\\approx$/-", "", f"{win1}/{tie1}/{loose1}", f"{win2}/{tie2}/{loose2}"]
data = pd.DataFrame(
    np.vstack([np.c_[problems, stats_HVNS, stats_GAHSS, stats_LGISS], summary]),
    columns=["Problem", "HVNS", "GAHSS", "LGISS"],
)
pd.set_option("display.max_rows", None)
print(data)
with open(f"stats_{metric}.txt", "w") as file:
    print(data, file=file)

caption = f"""We show the {metric} value (median and 10\\% - 90\\% quantile range) of the selected subset 
of the archive, averaged over 10 independent runs. The hypervolume Newton method (HVN) is compared to GAHSS
and LGISS using Mann–Whitney U test (with 5\\% significance level), where Holm-Sidak method is used to 
adjust the $p$-value for multiple testing. 
"""
data.to_latex(f"stats_{metric}.tex", index=False, caption=caption)
