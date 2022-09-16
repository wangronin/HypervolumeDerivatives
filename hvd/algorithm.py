import logging
from typing import Callable, Dict, List, Tuple, Union

import mpmath as mp
import numpy as np
from scipy.linalg import block_diag, cholesky
from scipy.spatial.distance import cdist

from . import mp_utils as mpu
from .hypervolume import hypervolume
from .hypervolume_derivatives import HypervolumeDerivatives
from .logger import get_logger
from .utils import non_domin_sort, set_bounds

__authors__ = ["Hao Wang"]

mp.mp.dps = 50


class HVN:
    """Hypervolume Newton method

    Newton-Raphson method applied to maximize the hypervolume indicator
    """

    def __init__(
        self,
        dim: int,
        n_objective: int,
        func: callable,
        jac: callable,
        hessian: callable,
        ref: Union[List[float], np.ndarray],
        mu: int = 5,
        h: Callable = None,
        h_jac: callable = None,
        h_hessian: callable = None,
        x0: np.ndarray = None,
        lower_bounds: Union[List[float], np.ndarray] = None,
        upper_bounds: Union[List[float], np.ndarray] = None,
        max_iters: Union[int, str] = np.inf,
        minimization: bool = True,
        xtol: float = 1e-3,
        HVtol: float = -np.inf,
        verbose: bool = True,
        **kwargs,
    ):
        """Hereafter, we use the following customized
        types to describe the usage:

        - Vector = List[float]
        - Matrix = List[Vector]

        Parameters
        ----------
        dim : int
            Dimensionality of the search space.
        obj_fun : Callable
            The objective function to be minimized.
        args: Tuple
            The extra parameters passed to function `obj_fun`.
        h : Callable, optional
            The equality constraint function, by default None.
        g : Callable, optional
            The inequality constraint function, by default None.
        x0 : Union[str, Vector, np.ndarray], optional
            The initial guess (by default None) which must fall between lower
            and upper bounds, if non-infinite values are provided for `lb` and
            `ub`. Note that, `x0` must be provided when `lb` and `ub` both
            take infinite values.
        sigma0 : Union[float], optional
            The initial step size, by default None
        C0 : Union[Matrix, np.ndarray], optional
            The initial covariance matrix which must be positive definite,
            by default None. Any non-positive definite input will be ignored.
        lb : Union[float, str, Vector, np.ndarray], optional
            The lower bound of search variables. When it is not a `float`,
            it must have the same length as `upper`, by default `-np.inf`.
        ub : Union[float, str, Vector, np.ndarray], optional
            The upper bound of search variables. When it is not a `float`,
            it must have the same length as `lower`, by default `np.inf`.
        ftarget : Union[int, float], optional
            The target value to hit, by default None.
        max_FEs : Union[int, str], optional
            Maximal number of function evaluations to make, by default `np.inf`.
        minimize : bool, optional
            To minimize or maximize, by default True.
        xtol : float, optional
            Absolute error in xopt between iterations that is acceptable for
            convergence, by default 1e-4.
        ftol : float, optional
            Absolute error in func(xopt) between iterations that is acceptable
            for convergence, by default 1e-4.
        n_restart : int, optional
            The maximal number of random restarts to perform when stagnation is
            detected during the run. The random restart can be switched off by
            setting `n_restart` to zero (the default value).
        verbose : bool, optional
            Verbosity of the output, by default False.
        logger : str, optional
            Name of the logger file, by default None, which turns off the
            logging behaviour.
        random_seed : int, optional
            The seed for pseudo-random number generators, by default None.
        """
        self.minimization = minimization
        self.dim_primal = dim
        self.n_objective = n_objective
        self.mu = mu  # the population size
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.ref = ref
        # parameters controlling stop criteria
        self.xtol = xtol
        self.HVtol = HVtol
        self.stop_dict: Dict = {}
        # the objective function, gradient, and the Hessian
        self.func: Callable = func
        self.h: Callable = h
        self.h_jac: Callable = h_jac
        self.h_hessian: Callable = h_hessian
        self.hypervolume_derivatives = HypervolumeDerivatives(
            self.dim_primal, self.n_objective, ref, func, jac, hessian, minimization=minimization
        )
        self.iter_count: int = 0
        self.max_iters = max_iters
        self.verbose: bool = verbose
        self.eps = 1e-3 * np.max(self.upper_bounds - self.lower_bounds)
        self._initialize(x0)
        self._init_logging_var()

    def _initialize(self, X0: np.ndarray):
        if X0 is not None:
            X0 = np.asarray(X0)
            assert np.all(X0 - self.lower_bounds >= 0)
            assert np.all(X0 - self.upper_bounds <= 0)
            assert X0.shape[0] == self.mu
        else:
            # sample `x` u.a.r. in `[lb, ub]`
            assert all(~np.isinf(self.lower_bounds)) & all(~np.isinf(self.upper_bounds))
            X0 = (
                np.random.rand(self.mu, self.dim_primal) * (self.upper_bounds - self.lower_bounds)
                + self.lower_bounds
            )  # (mu, d)

        self._max_HV = np.product(self.ref)
        # initialize dual variables
        if self.h is not None:
            v = self.h(X0[0, :])
            self.n_eq_cstr = 1 if isinstance(v, mp.mpf) else len(v)
            # to make the Hessian of Eq. constraints always a 3D tensor

            X0 = np.c_[X0, np.ones((self.mu, self.n_eq_cstr)) / self.mu]
        else:
            self.n_eq_cstr = 0

        self._get_primal_dual = lambda X: (X[:, : self.dim_primal], X[:, self.dim_primal :])
        self.dim = self.dim_primal + self.n_eq_cstr
        self.X = mpu.np2mp(X0)
        primal_vars = self._get_primal_dual(self.X)[0]
        self.Y = mpu.r_(*[self.func(primal_vars[i, :]) for i in range(primal_vars.rows)])  # (mu, n_objective)

    def _h_hessian(self, primal, dual):
        out = mp.zeros(self.dim_primal * self.n_eq_cstr, self.dim_primal * self.n_eq_cstr)
        for i, x in enumerate(self.h_hessian(primal)):
            out += x * dual[i]
        return out

    def _init_logging_var(self):
        """parameters for logging the history"""
        self.hist_Y: List[np.ndarray] = []
        self.hist_X: List[np.ndarray] = []
        self.hist_HV: List[float] = []
        self._delta_X: float = np.inf
        self._delta_Y: float = np.inf
        self._delta_HV: float = np.inf

        if self.h is not None:
            self.hist_G_norm: List[float] = []

        self.logger: logging.Logger = get_logger(
            logger_id=f"{self.__class__.__name__}",
            console=self.verbose,
        )

    @property
    def lower_bounds(self):
        return self._lower_bounds

    @lower_bounds.setter
    def lower_bounds(self, lb):
        self._lower_bounds = set_bounds(lb, self.dim_primal)

    @property
    def upper_bounds(self):
        return self._upper_bounds

    @upper_bounds.setter
    def upper_bounds(self, ub):
        self._upper_bounds = set_bounds(ub, self.dim_primal)

    @property
    def maxiter(self):
        return self._maxiter

    @maxiter.setter
    def maxiter(self, n: int):
        if n is None:
            self._maxiter = len(self.X) * 100

    def run(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        while not self.terminate():
            self.one_step()
            self.log()
        return self.X, self.Y, self.stop_dict

    def _compute_G(self, X: mp.matrix) -> mp.matrix:
        N = X.rows
        mud = int(N * self.dim_primal)
        primal_vars, dual_vars = self._get_primal_dual(X)
        out = self.hypervolume_derivatives.compute_gradient(mpu.mp2np(primal_vars))
        HVdX = out["HVdX"]
        dH = mpu.block_diag(*[self.h_jac(primal_vars[i, :]) for i in range(primal_vars.rows)])
        eq_cstr = mp.matrix([self.h(primal_vars[i, :]) for i in range(primal_vars.rows)]).T
        G = mpu.c_(*[HVdX + mpu.flatten(dual_vars) @ dH, eq_cstr])
        return mpu.c_(mpu.reshape(G[:mud], N, -1), mpu.reshape(G[mud:], N, -1))

    def _compute_netwon_step(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, np.ndarray]:
        N = X.rows
        primal_vars, dual_vars = self._get_primal_dual(X)
        out = self.hypervolume_derivatives.compute_hessian(mpu.mp2np(primal_vars), mpu.mp2np(Y))
        HVdX, HVdX2 = mpu.np2mp(out["HVdX"].ravel()).T, mpu.np2mp(out["HVdX2"])
        H, G = mpu.np2mp(HVdX2), mpu.np2mp(HVdX)

        if self.h is not None:  # with equality constraints
            mud = int(N * self.dim_primal)
            mup = int(N * self.n_eq_cstr)
            eq_cstr = mp.matrix([self.h(primal_vars[i, :]) for i in range(primal_vars.rows)]).T
            dH = mpu.block_diag(*[self.h_jac(primal_vars[i, :]) for i in range(primal_vars.rows)])
            ddH = mpu.block_diag(
                *[self._h_hessian(primal_vars[i, :], dual_vars[i, :]) for i in range(primal_vars.rows)]
            )  # (mu * dim, mu * dim)
            G = mpu.c_(*[HVdX + mpu.flatten(dual_vars) @ dH, eq_cstr]).T
            # NOTE: if the Hessian of the constraint is dropped, then quadratic convergence is gone
            H = mpu.r_(*[mpu.c_(HVdX2 + ddH, dH.T), mpu.c_(dH, mp.zeros(mup, mup))])

        step = -1 * (H**-1) @ G
        if self.h is not None:
            step = mpu.c_(mpu.reshape(step[:mud], N, -1), mpu.reshape(step[mud:], N, -1))
            G = mpu.c_(mpu.reshape(G[:mud], N, -1), mpu.reshape(G[mud:], N, -1))
            return dict(step=step, G=G)
        else:
            return dict(step=mpu.reshape(step, N, -1), G=mpu.reshape(G, N, -1))

    def one_step(self):
        self._check_XY()
        self.step = mp.zeros(self.mu, self.dim)
        self.step_size = mp.zeros(self.mu, self.mu)
        self.G = mp.zeros(self.mu, self.dim)

        # partition the approximation set to by feasibility
        if self.h is None:
            feasible_mask = np.array([True] * self.mu)
        else:
            primal_vars = self._get_primal_dual(self.X)[0]
            eq_cstr = mpu.mp2np(mp.matrix([self.h(primal_vars[i, :]) for i in range(primal_vars.rows)]).T)
            eq_cstr = eq_cstr.reshape(self.mu, -1)
            feasible_mask = np.all(np.isclose(eq_cstr, 0, atol=1e-3, rtol=1e-3), axis=1)
        # non-dominatd sorting of the feasible points
        if np.any(feasible_mask):
            feasible_idx = np.nonzero(feasible_mask)[0]
            partitions = non_domin_sort(
                mpu.mp2np(mpu.get_rows(self.Y, feasible_idx)), only_front_indices=True
            )
            partitions = {k: feasible_idx[v] for k, v in partitions.items()}
            partitions.update({0: np.sort(np.r_[partitions[0], np.nonzero(~feasible_mask)[0]])})
        else:
            partitions = {0: np.array(range(self.mu))}

        # compute the Newton direction for each partition
        for _, idx in partitions.items():
            out = self._compute_netwon_step(X=mpu.get_rows(self.X, idx), Y=mpu.get_rows(self.Y, idx))
            mpu.set_rows(self.step, idx, out["step"])
            mpu.set_rows(self.G, idx, out["G"])
            # backtracking line search with Armijo's condition for each point
            alpha = self._linear_search(
                mpu.get_rows(self.X, idx), mpu.get_rows(self.step, idx), G=mpu.get_rows(self.G, idx)
            )
            for i in idx:
                self.step_size[i, i] = alpha

        self.X = self.X + self.step_size @ self.step
        primal_vars = self._get_primal_dual(self.X)[0]
        self.Y = mpu.r_(*[self.func(primal_vars[i, :]) for i in range(primal_vars.rows)])  # (mu, n_objective)
        self.iter_count += 1

    def _linear_search(self, X: np.ndarray, step: np.ndarray, G: np.ndarray) -> float:
        """backtracking line search with Armijo's condition"""
        c = 1e-5
        N = len(X)
        primal_vars = mpu.mp2np(self._get_primal_dual(X)[0])
        normal_vectors = np.c_[np.eye(self.dim_primal * N), -1 * np.eye(self.dim_primal * N)]
        # calculate the maximal step-size
        dist = np.r_[
            np.abs(primal_vars.ravel() - np.tile(self.lower_bounds, N)),
            np.abs(np.tile(self.upper_bounds, N) - primal_vars.ravel()),
        ]
        v = mpu.mp2np(step[:, : self.dim_primal]).ravel() @ normal_vectors
        alpha = min(1, 0.25 * np.min(dist[v < 0] / np.abs(v[v < 0])))

        for _ in range(5):
            X_ = X + step * alpha
            if self.h is None:
                # TODO: support higher numerical precision in `HV`
                HV = self.hypervolume_derivatives.HV(mpu.mp2np(X))
                HV_ = self.hypervolume_derivatives.HV(mpu.mp2np(X_))
                inc = np.inner(G.ravel(), step.ravel())
                cond = HV_ - HV >= c * alpha * inc
            else:
                G_ = self._compute_G(X_)
                cond = mp.norm(G_) ** 2 <= (1 - 2 * c) * mp.norm(G) ** 2
            if cond:
                break
            else:
                if 11 < 2:
                    phi0 = HV if self.h is None else mp.norm(G) ** 2 / 2
                    phi1 = HV_ if self.h is None else mp.norm(G_) ** 2 / 2
                    phi0prime = inc if self.h is None else -mp.norm(G) ** 2
                    alpha = -phi0prime * alpha**2 / (phi1 - phi0 - phi0prime * alpha) / 2
                    # alpha *= tau
                if 1 < 2:
                    alpha *= 0.5
        else:
            self.logger.warn("Armijo's backtracking line search failed")
        return alpha

    def _check_XY(self):
        # get unique points: if some points converge to the same location
        primal_vars = mpu.mp2np(self._get_primal_dual(self.X)[0])
        Y = mpu.mp2np(self.Y)
        D = cdist(primal_vars, primal_vars)
        drop_idx_X = set([])
        for i in range(self.mu):
            if i not in drop_idx_X:
                drop_idx_X |= set(np.nonzero(D[i, :] < self.eps)[0]) - set([i])

        # get rid of weakly-dominated points
        drop_idx_Y = set([])
        for i in range(self.mu):
            if i not in drop_idx_Y:
                drop_idx_Y |= set(np.nonzero(np.isclose(Y[i, :], Y))[0]) - set([i])

        idx = list(set(range(self.mu)) - (drop_idx_X | drop_idx_Y))
        self.mu = len(idx)
        self.X = mpu.get_rows(self.X, idx)
        self.Y = mpu.get_rows(self.Y, idx)

    def log(self):
        HV = hypervolume(self.Y.tolist(), self.ref)
        self.hist_Y += [self.Y]
        self.hist_X += [self.X]
        self.hist_HV += [HV]

        if self.verbose:
            self.logger.info(f"iteration {self.iter_count} ---")
            self.logger.info(f"HV: {HV}")
            self.logger.info(f"step size: {np.diag(mpu.mp2np(self.step_size))}")

        if self.iter_count >= 1:
            try:
                # self._delta_X = np.mean(np.sqrt(np.sum((self.hist_X[-1] - self.hist_X[-2]) ** 2, axis=1)))
                # self._delta_Y = np.mean(np.sqrt(np.sum((self.hist_Y[-1] - self.hist_Y[-2]) ** 2, axis=1)))
                self._delta_HV = np.abs(self.hist_HV[-1] - self.hist_HV[-2])
            except:
                pass

        if self.h is not None:
            v = mp.norm(self.G)
            self.hist_G_norm += [v]
            self.logger.info(f"G norm: {float(v)}")

    def terminate(self) -> bool:
        if self.iter_count >= self.max_iters:
            self.stop_dict["iter_count"] = self.iter_count

        # if self._delta_HV < self.HVtol:
        #     self.stop_dict["HVtol"] = self._delta_HV
        #     self.stop_dict["iter_count"] = self.iter_count

        # if self._delta_X < self.xtol:
        #     self.stop_dict["xtol"] = self._delta_X
        #     self.stop_dict["iter_count"] = self.iter_count

        return bool(self.stop_dict)
