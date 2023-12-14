import re
from glob import glob

import numpy as np
import pandas
import pandas as pd

gen = 110
path = f"./DTLZ-gen{gen}/"
emoa = "NSGA-II"


def data_sanity_check(problem_name, emoa, run):
    component_file = f"{path}/{problem_name}_{emoa}_run_{run}_component_id_gen{gen}.csv"
    ref_label = pd.read_csv(component_file, header=None).values[0]
    n_cluster = len(np.unique(ref_label))
    n_cluster_ = len(
        [
            int(re.findall(r"ref_(\d+)_", s)[0])
            for s in glob(f"{path}/{problem_name}_{emoa}_run_{run}_ref_*_gen{gen}.csv")
        ]
    )
    if n_cluster_ != n_cluster_:
        print(f"{n_cluster_} ref. clusters found in files while {n_cluster} reported in {component_file}")

    # the load the final population from an EMOA
    x0 = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_x_gen{gen}.csv", header=None).values
    y0 = pd.read_csv(f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_y_gen{gen}.csv", header=None).values
    Y_label_file = f"{path}/{problem_name}_{emoa}_run_{run}_lastpopu_labels_gen{gen}.csv"
    Y_label = pd.read_csv(Y_label_file, header=None).values.ravel()
    Y_label = Y_label - 1
    # removing the outliers in `Y`
    idx = Y_label != -2
    n_outlier = np.sum(~idx)
    if n_outlier > 0:
        print(f"{n_outlier} outliers found in {Y_label_file}")

    x0 = x0[idx]
    y0 = y0[idx]
    Y_label = Y_label[idx]
    Y_idx = [np.nonzero(Y_label == i)[0] for i in range(n_cluster)]
    Y = [y0[idx] for idx in Y_idx]

    # load the reference set
    for i in range(n_cluster):
        ref_file = f"{path}/{problem_name}_{emoa}_run_{run}_ref_{i+1}_gen{gen}.csv"
        try:
            r = pd.read_csv(ref_file, header=None).values
            # if len(r) < len(Y[i]):
            # print(f"not enough reference points in {ref_file}")
        except pandas.errors.EmptyDataError:
            print(f"{ref_file} is empty")

        # downsample the reference; otherwise, initial clustering take too long
        eta_file = f"{path}/{problem_name}_{emoa}_run_{run}_eta_{i+1}_gen{gen}.csv"
        eta = pd.read_csv(eta_file, header=None).values.ravel()
        if np.any(np.isnan(eta)):
            print(f"`nan` values found in {eta_file}")

    # if the number of clusters of `Y` is more than that of the reference set
    if len(np.unique(Y_label)) > n_cluster:
        print(
            f"{problem_name}_{emoa}_run_{run}: the number of clusters of `Y` is more than that of the reference set"
        )


for problem_name in ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]:
    run_id = [
        int(re.findall(r"run_(\d+)_", s)[0])
        for s in glob(f"{path}/{problem_name}_{emoa}_run_*_lastpopu_x_gen{gen}.csv")
    ]
    for i in np.sort(run_id):
        data_sanity_check(problem_name, emoa, i)
