from typing import Tuple

import numpy as np
import pandas as pd

coefs = pd.read_csv("examples/subset_selection/ZDT2/coeff.csv", header=None).values
breaks = pd.read_csv("examples/subset_selection/ZDT2/knots.csv", header=None).values.ravel()


def pareto_front_approx(x: float) -> Tuple[float, float, float]:
    n = coefs.shape[0]
    interval = np.where(breaks <= x)[0][-1]
    interval_min = breaks[interval]
    interval_coefs = coefs[min(n - 1, interval), :]
    y = (
        interval_coefs[0] * (x - interval_min) ** 3
        + interval_coefs[1] * (x - interval_min) ** 2
        + interval_coefs[2] * (x - interval_min)
        + interval_coefs[3]
    )
    dy = (
        3 * interval_coefs[0] * (x - interval_min) ** 2
        + 2 * interval_coefs[1] * (x - interval_min)
        + interval_coefs[2]
    )
    d2y = 6 * interval_coefs[0] * (x - interval_min) + 2 * interval_coefs[1]
    return y, dy, d2y
