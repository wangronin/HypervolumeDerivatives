import numpy as np
import pandas as pd

gen = 110
emoa = "NSGA-II"
out = []
# for problem in ["ZDT1", "ZDT2", "ZDT3", "ZDT4",
dfs = []
for problem in ["ZDT6"]:
    data1 = pd.read_csv(f"./results/{problem}-DpN-{emoa}-{gen}.csv")
    data2 = pd.read_csv(f"./results/{problem}-{emoa}-{gen}.csv")
    df1 = data1.loc[:, ["IGD"]]
    df1.insert(0, "dataset_name", problem)
    df1.insert(0, "classifier_name", "DpN")
    df2 = data2.loc[:, ["IGD"]]
    df2.insert(0, "dataset_name", problem)
    df2.insert(0, "classifier_name", emoa)
    dfs.append(pd.concat([df1, df2], axis=0))

    df1 = data1.apply(lambda x: f"{np.median(x):.4e} +/- {np.std(x) / np.sqrt(len(x)):.4e}", axis=0)
    df2 = data2.apply(lambda x: f"{np.median(x):.4e} +/- {np.std(x) / np.sqrt(len(x)):.4e}", axis=0)
    out.append(
        pd.DataFrame(
            [[problem, "DpN"] + df1.values.tolist(), [problem, "NSGA-II"] + df2.values.tolist()],
            columns=["problem", "algorithm"] + df1.index.to_list(),
        )
    )

df = pd.concat(dfs, axis=0)
df.rename(columns={"IGD": "accuracy"}, inplace=True)
df.to_csv("example.csv", index=False)

out = pd.concat(out, axis=0)
print(out)
print(out.to_latex(index=False))
