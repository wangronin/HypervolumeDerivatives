from typing import List, Tuple, Union

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar


def read_bspline_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray, int]:
    points: List[float] = []
    knots: List[float] = []
    degree: int = None
    mode: str = "points"

    with open(filepath, "r") as file:
        for line_num, line in enumerate(file):
            line = line.strip()
            if not line:
                continue  # skip empty lines

            # Skip header line
            if line_num == 0 and line.lower() == "x,y,z":
                continue

            # Change mode based on section headers if any
            if line.lower().startswith("knot"):
                mode = "knots"
                continue
            if line.lower().startswith("degree"):
                mode = "degree"
                continue

            if mode == "points":
                parts = line.split(",")
                if len(parts) == 3:
                    points.append([float(p) for p in parts])
            elif mode == "knots":
                parts = line.split(",")
                for p in parts:
                    if p.strip():
                        knots.append(float(p.strip()))
            elif mode == "degree":
                degree = int(float(line))  # sometimes degree may be written as 3.0 instead of 3

    points = np.array(points)
    knots = np.array(knots)
    return points, knots, degree


# TODO: we should be able to get it for free from the fitting of B-spline
def get_preimage(y: np.ndarray, bs: callable) -> np.ndarray:
    f = lambda t: np.sum((bs.objective(t) - y) ** 2)
    result = minimize_scalar(f, bounds=(0, 1), method="bounded", options=dict(xatol=1e-10, maxiter=1000))
    return result.x


class PiecewiseBS:
    def __init__(self, files: List[str]) -> None:
        self._bsplines = self.sort_splines(np.array([BS(*read_bspline_csv(file)) for file in files]))
        self.N = len(self._bsplines)
        self.intervals = np.linspace(0, 1, self.N + 1)
        self.size = 1 / self.N

    def sort_splines(self, splines: List[BSpline]) -> List[BSpline]:
        indices = np.argsort([bs.objective(0)[0] for bs in splines])
        return splines[indices]

    def get_index(self, t: float) -> int:
        """get the index of the B-spline to which the input parameter `t` belongs

        Args:
            t (float): the parameter value

        Returns:
            int: the index of the B-spline that `t` belongs to
        """
        k = np.floor(t / (self.size))
        if t == 1.0:
            k = self.N - 1
        return int(k)

    def __func__(self, t: Union[np.ndarray, float], type: str):
        k = self.get_index(t)
        return getattr(self._bsplines[k], type)(self.N * (t - self.intervals[k]))

    def objective(self, t: Union[np.ndarray, float]) -> np.ndarray:
        return self.__func__(t, "objective")

    def jacobian(self, t: Union[np.ndarray, float]) -> np.ndarray:
        return self.__func__(t, "jacobian")

    def hessian(self, t: Union[np.ndarray, float]) -> np.ndarray:
        return self.__func__(t, "hessian")


class BS:
    def __init__(self, control_points: np.ndarray, knots: np.ndarray, degree: int) -> None:
        self._bs = BSpline(t=knots, c=control_points[:, 0:2], k=degree, extrapolate=True)
        self._bs_jac = self._bs.derivative()
        self._bs_hess = self._bs_jac.derivative()

    def objective(self, t: Union[np.ndarray, float]) -> np.ndarray:
        if isinstance(t, np.ndarray):
            t = t[0]  # ensure the output shape is (2,)
        return self._bs(t)

    def jacobian(self, t: Union[np.ndarray, float]) -> np.ndarray:
        if isinstance(t, np.ndarray):
            t = t[0]  # ensure the output shape is (2,1)
        # we set the Jacobian of the boundary point to zero to make it remain stationary
        return np.zeros((2, 1)) if t == 0 or t == 1 else self._bs_jac(t).reshape(-1, 1)

    def hessian(self, t: Union[np.ndarray, float]) -> np.ndarray:
        if isinstance(t, np.ndarray):
            t = t[0]  # ensure the output shape is (2, 1, 1)
        # we set the Hessian of the boundary point to zero to make it remain stationary
        return np.zeros((2, 1, 1)) if t == 0 or t == 1 else self._bs_hess(t).reshape(2, 1, 1)
