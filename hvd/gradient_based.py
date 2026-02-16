import logging
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist

from .base import State
from .utils import get_logger, set_bounds


class Lara:

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        func: callable,
        jac: callable,
        xl: Union[List[float], np.ndarray],
        xu: Union[List[float], np.ndarray],
        N: int = 5,
        X0: np.ndarray = None,
        max_iters: Union[int, str] = np.inf,
        xtol: float = 1e-3,
        verbose: bool = True,
        metrics: Dict[str, Callable] = dict(),
    ):
        self.dim: int = n_var  # the number of primal variables
        self.n_obj: int = n_obj  # the number of objectives
        self.N: int = N  # the population size
        self.xl: np.ndarray = xl
        self.xu: np.ndarray = xu
        self.state: State = State(self.dim, 0, 0, func, jac)
        self._initialize(X0)
        self._set_logging(verbose)
        self.xtol: float = xtol
        self.max_iters: int = self.N * 10 if max_iters is None else max_iters
        self.stop_dict: Dict[str, float] = {}
        self.metrics: Dict[str, Callable] = metrics
        self.verbose: bool = verbose
        self.eps: float = 1e-3 * np.max(self.xu - self.xl)

    def _initialize(self, X0: np.ndarray):
        if X0 is not None:
            X0 = np.asarray(X0)
            X0 = np.clip(X0, self.xl, self.xu)
            # NOTE: ad-hoc solution for CF2 and IDTLZ1 since the Jacobian on the box boundary is not defined
            # on the decision boundary or the local Hessian is ill-conditioned.
            if 11 < 2:
                X0 = np.clip(X0 - self.xl, 1e-2, 1) + self.xl
                X0 = np.clip(X0 - self.xu, -1, -1e-2) + self.xu
            self.N = len(X0)
        else:
            # sample `x` u.a.r. in `[lb, ub]`
            assert self.N is not None
            assert all(~np.isinf(self.xl)) & all(~np.isinf(self.xu))
            X0 = np.random.rand(self.N, self.dim) * (self.xu - self.xl) + self.xl  # (mu, dim_primal)
        # initialize the state variables
        self.state.update(X0)  # (mu, dim)
        self.iter_count: int = 0

    def _set_logging(self, verbose: bool):
        """parameters for logging the history"""
        self.verbose: bool = verbose
        self.curr_indicator_value: float = None
        self.history_Y: List[np.ndarray] = []
        self.history_X: List[np.ndarray] = []
        self.history_indicator_value: List[float] = []
        self.history_R_norm: List[float] = []
        self.history_metrics: Dict[str, List] = defaultdict(list)
        self.history_medoids: Dict[int, List] = defaultdict(list)
        self.logger: logging.Logger = get_logger(logger_id=f"{self.__class__.__name__}", console=self.verbose)

    @property
    def xl(self):
        return self._xl

    @xl.setter
    def xl(self, lb):
        self._xl = set_bounds(lb, self.dim)

    @property
    def xu(self):
        return self._xu

    @xu.setter
    def xu(self, ub):
        self._xu = set_bounds(ub, self.dim)

    def run(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        while not self.terminate():
            self.iteration()
            self.log()
        return self.state.primal, self.state.Y, self.stop_dict

    def iteration(self):
        # check for anomalies in `X` and `Y`
        self._check_points_uniqueness()
        # first compute the current indicator value
        self.step_size = np.ones(self.N)
        self.step = self._compute_step(self.state)
        # constrain the search steps within the search box
        self.step, max_step_size = self._handle_box_constraint(self.step, self.state)
        self.step_size = np.clip(self.step_size, 0, max_step_size)
        # Newton iteration and evaluation
        self.state.update(self.state.X + self.step * self.step_size.reshape(-1, 1))

    def log(self):
        # TODO: maybe we should log the initial population
        self.iter_count += 1
        self.history_X += [self.state.primal.copy()]
        self.history_Y += [self.state.Y.copy()]
        self.history_R_norm += [np.median(np.linalg.norm(self.step, axis=1))]
        # compute the performance metrics
        for name, func in self.metrics.items():
            self.history_metrics[name].append(func.compute(Y=self.state.Y))
        if self.verbose:
            self.logger.info(f"iteration {self.iter_count} ---")
            self.logger.info(f"step size: {self.step_size.ravel()}")
            self.logger.info(f"R norm: {self.history_R_norm[-1]}")

    def terminate(self) -> bool:
        if self.iter_count >= self.max_iters:
            self.stop_dict["iter_count"] = self.iter_count

        return bool(self.stop_dict)

    def _compute_step(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        R = np.zeros((state.N, state.n_var))
        for i in range(state.N):
            v = state.J[i]
            # TODO: FIXIT!!
            R[i] = -1 * np.sum([v[k] / np.linalg.norm(v[k]) for k in range(self.n_obj)], axis=0)
        return R

    def _handle_box_constraint(self, step: np.ndarray, state: State) -> Tuple[np.ndarray, np.ndarray]:
        primal_vars, step_primal = state.primal, step[:, : self.dim]
        # distance to the lower and upper boundary per dimension
        dist_xl, dist_xu = np.abs(primal_vars - self.xl), np.abs(self.xu - primal_vars)
        # if a point on a lower boundary and its has a negative step on this dimension
        r, c = np.nonzero(np.isclose(dist_xl, 0, atol=1e-10, rtol=1e-10) & (step_primal < 0))
        step_primal[r, c] = 0
        # if a point on an upuper boundary and its has a positive step on this dimension
        r, c = np.nonzero(np.isclose(dist_xu, 0, atol=1e-10, rtol=1e-10) & (step_primal > 0))
        step_primal[r, c] = 0
        s = np.c_[dist_xl / step_primal, -dist_xu / step_primal]
        s[s >= 0] = np.inf
        s = np.minimum(1, np.nanmin(np.abs(s), axis=1))
        step[:, : self.dim] = step_primal
        return step, min(s)

    def _check_points_uniqueness(self):
        """check uniqueness of decision and objective points.
        if two points are identical up to a high precision, then we remove one of them.
        """
        primal_vars = self.state.primal
        D = cdist(primal_vars, primal_vars)
        drop_idx_X = set([])
        for i in range(self.N):
            if i not in drop_idx_X:
                drop_idx_X |= set(np.nonzero(np.isclose(D[i, :], 0, rtol=self.eps))[0]) - set([i])

        # TODO: get rid of weakly-dominated points
        drop_idx_Y = set([])
        drop_idx = drop_idx_X | drop_idx_Y
        if len(drop_idx) > 0:
            self.logger.info(f"{len(drop_idx)} points are removed due to duplication")
        idx = list(set(range(self.N)) - drop_idx)
        self.state = self.state[idx]
        self.N = self.state.N
