import logging
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from autograd import hessian, jacobian
from scipy.linalg import block_diag, cho_solve, cholesky
from scipy.optimize.linesearch import line_search_wolfe1, line_search_wolfe2
from scipy.spatial.distance import cdist

from .hypervolume import hypervolume
from .hypervolume_derivatives import HypervolumeDerivatives
from .logger import get_logger
from .utils import handle_box_constraint, non_domin_sort, set_bounds

__authors__ = ["Hao Wang"]


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
        self.dim = dim
        self.n_objective = n_objective
        self.mu = mu  # the population size
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
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
            self.dim, self.n_objective, ref, func, jac, hessian, minimization=minimization
        )
        # set the initial search point
        self.X = x0
        self.ref = ref
        self.iter_count: int = 0
        self.max_iters = max_iters
        self.verbose: bool = verbose
        self.eps = 1e-3 * np.max(self.upper_bounds - self.lower_bounds)
        self._initialize(x0)
        self._init_logging_var()

        self._cumulation = np.zeros(self.mu)
        self._step_size = 0.05 * np.ones(self.mu) * np.max((self.upper_bounds - self.lower_bounds))
        self._pre_step = np.zeros((self.mu, self.dim))
        self._grad_norm = np.zeros(self.mu)
        # self._cumulation = 0
        # self._step_size = 0.001 * np.max((self.upper_bounds - self.lower_bounds))
        # self._pre_step = np.zeros((self.mu, self.dim))
        # self._grad_norm = 0

    def _initialize(self, X0):
        if X0 is not None:
            X0 = np.asarray(X0)
            assert np.all(X0 - self.lower_bounds >= 0)
            assert np.all(X0 - self.upper_bounds <= 0)
            assert X0.shape[0] == self.mu
        else:
            # sample `x` u.a.r. in `[lb, ub]`
            assert all(~np.isinf(self.lower_bounds)) & all(~np.isinf(self.upper_bounds))
            X0 = (
                np.random.rand(self.mu, self.dim) * (self.upper_bounds - self.lower_bounds)
                + self.lower_bounds
            )  # (mu, d)
        self.X = X0
        self.Y = np.array([self.func(x) for x in self.X])  # (mu, n_objective)
        # initialize dual variables
        if self.h is not None:
            v = self.h(self.X[0, :])
            self.n_eq_cstr = 1 if isinstance(v, float) else len(v)
            # TODO: what is the best way to initialize lambdas
            self.dual_vars = np.ones((self.mu, self.n_eq_cstr)) / self.mu  # of size (mu, p)
            self.eq_cstr_value = np.atleast_2d([[self.h(_)] for _ in self.X])
            # to make the Hessian of Eq. constraints always a 3D tensor
            self._h_hessian = lambda x: self.h_hessian(x).reshape(self.n_eq_cstr, self.dim, -1)

        (
            self._box_constraint,
            self._box_constraint_jacobian,
            self._box_constraint_hessian,
        ) = self._set_box_constraint()

        self._max_HV = np.product(self.ref)
        # self.t = 1e-5

    def _set_box_constraint(self):
        # set the box constraint function
        def _box_constraint(x):
            # return np.concatenate(
            #     [
            #         np.min(-np.exp(self.lower_bounds - x) + 1, 0),
            #         np.min(-np.exp(x - self.upper_bounds) + 1, 0),
            #     ]
            # )
            return np.sum(
                1 / (1 + np.exp(self.lower_bounds - x)) - 1 + 1 / (1 + np.exp(x - self.upper_bounds)) - 1
            )

        _box_constraint_jacobian = jacobian(_box_constraint)
        _box_constraint_hessian = hessian(_box_constraint)
        # def _box_constraint_jacobian(x):
        #     return np.exp(self.lower_bounds - x) - np.exp(x - self.upper_bounds)

        # def _box_constraint_hessian(x):
        #     return np.diag(-1.0 * np.exp(self.lower_bounds - x) - np.exp(x - self.upper_bounds))

        return _box_constraint, _box_constraint_jacobian, _box_constraint_hessian

    def _init_logging_var(self):
        """parameters for logging the history"""
        self.hist_Y: List[np.ndarray] = []
        self.hist_X: List[np.ndarray] = []
        self.hist_HV: List[float] = []
        self._delta_X: float = np.inf
        self._delta_Y: float = np.inf
        self._delta_HV: float = np.inf

        if self.h is not None:
            self.hist_dual_vars: List[np.ndarray] = []
            self.hist_eq_cstr: List[np.ndarray] = []
            self.hist_eq_cstr_norm: List[float] = []

        self.logger: logging.Logger = get_logger(
            logger_id=f"{self.__class__.__name__}",
            console=self.verbose,
        )

    @property
    def lower_bounds(self):
        return self._lower_bounds

    @lower_bounds.setter
    def lower_bounds(self, lb):
        self._lower_bounds = set_bounds(lb, self.dim)

    @property
    def upper_bounds(self):
        return self._upper_bounds

    @upper_bounds.setter
    def upper_bounds(self, ub):
        self._upper_bounds = set_bounds(ub, self.dim)

    @property
    def maxiter(self):
        return self._maxiter

    @maxiter.setter
    def maxiter(self, n: int):
        if n is None:
            self._maxiter = len(self.X) * 100

    def run(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        while not self.terminate():
            self.step()
            # self.log()
        return self.X, self.Y, self.stop_dict

    def restart(self):
        # self.logger.info("restarting... ")
        self.X = None
        self.stop_dict = {}
        self.n_restart -= 1

    def _compute_netwon_step(self, X: np.ndarray, Y: np.ndarray, dual_vars: np.ndarray = None) -> Dict:
        out = self.hypervolume_derivatives.compute(X, Y)
        # normalize the HV value to avoid large gradient values
        dHV, ddHV = out["HVdX"] / self._max_HV, out["HVdX2"] / self._max_HV
        # dHV, ddHV = out["HVdX"], out["HVdX2"]

        if self.h is not None and dual_vars is not None:  # with equality constraints
            mud = int(X.shape[0] * self.dim)
            mup = int(X.shape[0] * self.n_eq_cstr)
            H = np.array([self.h(_) for _ in X]).reshape(X.shape[0], -1)  # (mu, p)
            dH = block_diag(*[self.h_jac(x) for x in X])  # (mu * p, dim)
            ddH = block_diag(
                *[np.einsum("ijk,i->jk", self._h_hessian(x), dual_vars[i]) for i, x in enumerate(X)]
            )  # (mu * dim, mu * dim)
            G = np.concatenate([dHV.ravel() + dual_vars.ravel() @ dH, H.ravel()])
            dG = np.concatenate(
                [
                    np.concatenate([ddHV, dH.T], axis=1),
                    np.concatenate([dH, np.zeros((mup, mup))], axis=1),
                ],
            )
            # TODO: perhaps using Cholesky decomposition to speed up
            newton_step = np.linalg.solve(dG, G)  # compute the Netwon's step
            return dict(step_X=newton_step[0:mud], step_dual=newton_step[mud:], grad=dHV.ravel(), H=H)
        else:

            # dB = np.array([self._box_constraint_jacobian(x) for x in X]).ravel()
            # ddB = block_diag(*[self._box_constraint_hessian(x) for x in X])
            G = dHV
            dG = ddHV

            # pre-condition the Hessian
            beta = 1e-3
            v = np.min(np.diag(-dG))
            tau = 0 if v > 0 else -v + beta
            I = np.eye(dG.shape[0])
            for _ in range(50):
                try:
                    L = cholesky(-dG + tau * I, lower=True)
                    break
                except:
                    tau = max(1.3 * tau, beta)
            else:
                self.logger.warn("Armijo's backtracking line search failed")

            newton_step = cho_solve((L, True), G.ravel())
            # newton_step = np.linalg.solve(dG, G.ravel())
            # assert that `newton_step` is always an ascending direction
            # if not np.inner(newton_step, dHV) >= 0:
            # breakpoint()

            return dict(step_X=newton_step, grad=dHV.ravel())

    def step(self):
        def f(x):
            return self.hypervolume_derivatives.HV(x) / self._max_HV

        # get unique points: if some points converge to the same location
        D = cdist(self.X, self.X)
        drop_idx_X = set([])
        for i in range(self.mu):
            if i not in drop_idx_X:
                drop_idx_X |= set(np.nonzero(D[i, :] < self.eps)[0]) - set([i])

        # get rid of weakly-dominated points
        drop_idx_Y = set([])
        for i in range(self.mu):
            if i not in drop_idx_Y:
                drop_idx_Y |= set(np.nonzero(np.isclose(self.Y[i, :], self.Y))[0]) - set([i])

        idx = list(set(range(self.mu)) - (drop_idx_X | drop_idx_Y))
        self.mu = len(idx)
        self.X = self.X[idx, :]
        self.Y = self.Y[idx, :]

        self._cumulation = self._cumulation[idx]
        self._step_size = self._step_size[idx]
        self._pre_step = self._pre_step[idx, :]
        if self.h is not None:
            self.dual_vars = self.dual_vars[idx, :]
            self.eq_cstr_value = self.eq_cstr_value[idx, :]
            dual_step = np.zeros((self.mu, self.n_eq_cstr))

        self.log()

        step = np.zeros((self.mu, self.dim))
        grad = np.zeros((self.mu, self.dim))
        # partition approximation set to anti-chains
        fronts = non_domin_sort(self.Y, only_front_indices=True)
        self.step_size = np.zeros(self.mu)
        # Newton-Raphson method for each front and compute the HV newton direction
        for _, idx in fronts.items():
            N = len(idx)
            out = self._compute_netwon_step(
                X=self.X[idx, :],
                Y=self.Y[idx, :],
                dual_vars=self.dual_vars[idx, :] if self.h is not None else None,
            )
            step[idx, :] = out["step_X"].reshape(N, -1)
            grad[idx, :] = out["grad"].reshape(N, -1)
            if self.h is not None:
                dual_step[idx, :] = out["step_dual"].reshape(N, -1)
                self.eq_cstr_value[idx, :] = out["H"]

            c = 1e-2
            for i in idx:
                alpha = 1
                for _ in range(40):
                    X_ = self.X[idx, :].copy()
                    X_ += alpha * step[i, :]
                    # Armijo's condition
                    if f(X_) - f(self.X[idx, :]) >= c * alpha * np.inner(
                        grad[i, :].ravel(), step[i, :].ravel()
                    ):
                        break
                    else:
                        alpha *= 0.5
                else:
                    self.logger.warn("Armijo's backtracking line search failed")

                self.X[i, :] += alpha * step[i, :]
                self.step_size[i] = alpha

        self.logger.info(self.step_size)
        self._grad_norm = np.linalg.norm(grad.ravel())
        step_norm = np.sqrt(np.sum(step**2, axis=1))
        self.logger.info(f"step norm {step_norm}")

        # c = 1e-5
        # self.step_size = np.zeros(self.mu)
        # for i in range(self.mu):
        #     alpha = 1
        #     for _ in range(20):
        #         x_ = self.X.copy()
        #         x_[i, :] += alpha * step[i, :]
        #         # Armijo's condition
        #         if f(x_) - f(self.X) >= c * alpha * np.inner(grad[i, :], step[i, :]):
        #             break
        #         else:
        #             alpha *= 0.5
        #     else:
        #         self.logger.warn("Armijo's backtracking line search failed")

        #     self.X[i, :] += alpha * step[i, :]
        #     self.step_size[i] = alpha

        # handle the box constraints
        self.X = handle_box_constraint(self.X, self.lower_bounds, self.upper_bounds)
        # self.X -= step
        # if self.h is not None:
        #     self.dual_vars -= dual_step

        # for i in range(self.mu):
        #     _step = step[i, :]
        #     _norm = np.linalg.norm(_step)
        #     if not np.isclose(_norm, 0):
        #         _step /= _norm

        #     self._cumulation[i] = (1 - c) * self._cumulation[i] + c * np.inner(_step, self._pre_step[i, :])
        #     self._step_size[i] *= np.exp(self._cumulation[i] * alpha)

        #     self.X[i, :] -= self._step_size[i] * _step
        #     self.X[i, :] = np.clip(self.X[i, :], 1.2 * self.lower_bounds, 0.9 * self.upper_bounds)
        #     self._pre_step[i, :] = _step

        # self.X -= step
        # # normalize the newton step and the gradient
        # grad = out["grad"].reshape(N, -1)
        # _norm = np.sqrt(np.sum(grad**2, axis=1))
        # self._grad_norm[idx] = _norm
        # _idx = ~np.isclose(_norm, 0)
        # grad[_idx, :] /= _norm[_idx].reshape(-1, 1)

        # step = out["step_X"].reshape(N, -1)
        # _norm = np.sqrt(np.sum(step**2, axis=1))
        # _idx = ~np.isclose(_norm, 0)
        # step[_idx, :] /= _norm[_idx].reshape(-1, 1)

        # self._cumulation[idx] = (1 - c) * self._cumulation[idx] + c * np.array(
        #     [np.inner(grad[i, :], self._pre_step[_, :]) for i, _ in enumerate(idx)]
        # )
        # self._step_size[idx] *= np.exp((self._cumulation[idx]) * alpha)
        # print(self._step_size)
        # _idx = np.nonzero(_norm > _range)[0]
        # if len(_idx) > 0:
        # breakpoint()
        # step[_idx, :] = 2 * step[_idx, :] / _norm[_idx].reshape(-1, 1)
        # self.X[idx, :] -= self._step_size[idx].reshape(-1, 1) * step
        # if self.h is not None:
        #     self.dual_vars[idx, :] -= out["step_dual"].reshape(N, -1)
        #     self.eq_cstr_value[idx, :] = out["H"]

        # self._pre_step[idx, :] = step

        # evaluation
        self.Y = np.array([self.func(x) for x in self.X])
        self.iter_count += 1

    def log(self):
        HV = hypervolume(self.Y, self.ref)
        # print(np.sum(self.Y, axis=1))
        self.hist_Y += [self.Y.copy()]
        self.hist_X += [self.X.copy()]
        self.hist_HV += [HV]

        if self.verbose:
            self.logger.info(f"iteration {self.iter_count} ---")
            self.logger.info(f"X {self.X.ravel()}")
            self.logger.info(f"HV {HV}")
            # self.logger.info(f"step size {self._step_size}")
            # self.logger.info(f"cumulation {self._cumulation}")
            self.logger.info(f"grad norm {self._grad_norm}")
            # if self.iter_count >= 1:
            # self.logger.info(f"step norm {np.linalg.norm(self._step.ravel())}")

        if self.iter_count >= 1:
            try:
                self._delta_X = np.mean(np.sqrt(np.sum((self.hist_X[-1] - self.hist_X[-2]) ** 2, axis=1)))
                self._delta_Y = np.mean(np.sqrt(np.sum((self.hist_Y[-1] - self.hist_Y[-2]) ** 2, axis=1)))
                self._delta_HV = np.abs(self.hist_HV[-1] - self.hist_HV[-2])
            except:
                pass

        if self.h is not None:
            eq_cstr_norm = np.mean(np.linalg.norm(self.eq_cstr_value, axis=1))
            self.logger.info(f"dual variables: {self.dual_vars.ravel()}")
            self.logger.info(f"Equality constraints: {self.eq_cstr_value.ravel()}")
            self.hist_dual_vars += [self.dual_vars]
            self.hist_eq_cstr += [self.eq_cstr_value]
            self.hist_eq_cstr_norm += [eq_cstr_norm]

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
