from typing import Tuple

import numpy as np
import pandas as pd

coefs1 = pd.read_csv("examples/subset_selection/TwoOnOne/coeff_1.csv", header=None).values
breaks1 = pd.read_csv("examples/subset_selection/TwoOnOne/knots_1.csv", header=None).values.ravel()

coefs2 = pd.read_csv("examples/subset_selection/TwoOnOne/coeff_2.csv", header=None).values
breaks2 = pd.read_csv("examples/subset_selection/TwoOnOne/knots_2.csv", header=None).values.ravel()


def pareto_front_approx(x: float) -> Tuple[float, float, float]:
    if x >= 7.66439406746696:
        return component1(x)
    else:
        return component2(x)


def component1(x: float) -> Tuple[float, float, float]:
    x = np.clip(x, breaks1.min(), breaks1.max())
    interval = np.where(breaks1 <= x)[0][-1]
    interval_min = breaks1[interval]
    interval_coefs = coefs1[min(30, interval), :]

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


def component2(x: float) -> Tuple[float, float, float]:
    x = np.clip(x, breaks2.min(), breaks2.max())
    interval = np.where(breaks2 <= x)[0][-1]
    interval_min = breaks2[interval]
    interval_coefs = coefs2[min(30, interval), :]

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
