import numpy as np
import pandas as pd

out = []
for problem in ["ZDT1", "ZDT2", "ZDT3", "ZDT4"]:
    data1 = pd.read_csv(f"./results/{problem}-DpN.csv")
    data2 = pd.read_csv(f"./results/{problem}-NSGA-II.csv")
    df1 = data1.apply(lambda x: f"{np.median(x):.4e} +/- {np.std(x) / np.sqrt(len(x)):.4e}", axis=0)
    df2 = data2.apply(lambda x: f"{np.median(x):.4e} +/- {np.std(x) / np.sqrt(len(x)):.4e}", axis=0)
    out.append(
        pd.DataFrame(
            [[problem, "DpN"] + df1.values.tolist(), [problem, "NSGA-II"] + df2.values.tolist()],
            columns=["problem", "algorithm"] + df1.index.to_list(),
        )
    )

out = pd.concat(out, axis=0)
print(out)
print(out.to_latex(index=False))
