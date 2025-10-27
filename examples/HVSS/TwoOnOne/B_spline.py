from typing import Union

import numpy as np
from scipy.interpolate import BSpline

coeff1 = np.array(
    [
        [7.731731294582435, 2.720579185560172],
        [8.075943083432229, 2.105900723602403],
        [8.883235693388814, 1.90215517011007],
        [9.371500282813152, 1.6060883345349337],
        [10.122694973338746, 1.438270844610269],
        [10.74806349925855, 1.2548577451719714],
        [12.159522532645433, 0.9372704445468596],
        [13.564246589951187, 0.6808771054325522],
        [14.317456535780622, 0.5687160675481976],
        [15.052830562550774, 0.45768986306030457],
        [15.804941221199936, 0.364909202650419],
        [17.21761799032171, 0.21118526833626236],
        [18.6380805900799, 0.09799145864919304],
        [19.305668073677257, 0.05865660531871616],
        [19.977915636474833, 0.025133439137902242],
        [20.646690580959877, 0.008752978695458633],
        [21.317898814244383, 7.182752547391587e-05],
        [21.988287787047025, 1.8499667480911051e-06],
    ]
)
k1 = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.32442190380805,
        0.32442190380805,
        0.32442190380805,
        0.32442190380805,
        0.32442190380805,
        0.68168736172981,
        0.68168736172981,
        0.68168736172981,
        0.68168736172981,
        0.68168736172981,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
)
coeff2 = np.array(
    [
        [6.834181184714719, 9.028524993531438],
        [6.83800472648028, 8.852679376897232],
        [6.868543688200454, 8.679427392638019],
        [6.907402875674087, 8.50561820246261],
        [6.974722432748125, 8.342430644038542],
        [7.113768004162804, 8.05169556704464],
        [7.289938747190306, 7.78180300730777],
        [7.38290824986341, 7.6685681571053905],
        [7.462654148242079, 7.5457155069937425],
        [7.567579096126696, 7.442328025353268],
        [7.664306258633038, 7.333340222127541],
    ]
)
k2 = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5458305491446442,
        0.5458305491446442,
        0.5458305491446442,
        0.5458305491446442,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
)
bs1 = BSpline(t=k1, c=coeff1[:, 0:2], k=7, extrapolate=False)
bs_jac1 = bs1.derivative()
bs_hess1 = bs_jac1.derivative()

bs2 = BSpline(t=k2, c=coeff2[:, 0:2], k=6, extrapolate=False)
bs_jac2 = bs2.derivative()
bs_hess2 = bs_jac2.derivative()


def objective1(t: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(t, np.ndarray):
        t = t[0]  # ensure the output shape is (2,)
    return bs1(t)


def jacobian1(t: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(t, np.ndarray):
        t = t[0]  # ensure the output shape is (2,1)
    # whenever a point is on the boundary, we set the corresponding Jacobian to zero
    # to make the point stationary
    if t == 0 or t == 1:
        return np.zeros((2, 1))
    else:
        return bs_jac1(t).reshape(-1, 1)


def hessian1(t: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(t, np.ndarray):
        t = t[0]  # ensure the output shape is (2, 1, 1)
    # whenever a point is on the boundary, we set the corresponding Jacobian to zero
    # to make the point stationary
    if t == 0 or t == 1:
        return np.zeros((2, 1, 1))
    else:
        return bs_hess1(t).reshape(2, 1, 1)


def objective2(t: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(t, np.ndarray):
        t = t[0]  # ensure the output shape is (2,)
    return bs2(t)


def jacobian2(t: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(t, np.ndarray):
        t = t[0]  # ensure the output shape is (2,1)
    # whenever a point is on the boundary, we set the corresponding Jacobian to zero
    # to make the point stationary
    if t == 0 or t == 1:
        return np.zeros((2, 1))
    else:
        return bs_jac2(t).reshape(-1, 1)


def hessian2(t: Union[np.ndarray, float]) -> np.ndarray:
    if isinstance(t, np.ndarray):
        t = t[0]  # ensure the output shape is (2, 1, 1)
    # whenever a point is on the boundary, we set the corresponding Jacobian to zero
    # to make the point stationary
    if t == 0 or t == 1:
        return np.zeros((2, 1, 1))
    else:
        return bs_hess2(t).reshape(2, 1, 1)
