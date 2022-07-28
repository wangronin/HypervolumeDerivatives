import logging
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.linalg import block_diag

from .hypervolume import hypervolume
from .hypervolume_derivatives import HypervolumeDerivatives
from .logger import get_logger
from .utils import non_domin_sort, set_bounds

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
        verbose: bool = False,
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
        self._initialize(x0)
        self._init_logging_var()

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
            self.eq_cstr_value = np.atleast_2d([self.h(_) for _ in self.X])
            # to make the Hessian of Eq. constraints always a 3D tensor
            self._h_hessian = lambda x: self.h_hessian(x).reshape(self.n_eq_cstr, self.dim, -1)

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
            self.log()
            self.step()
        return self.X, self.Y, self.stop_dict

    def restart(self):
        # self.logger.info("restarting... ")
        self.X = None
        self.stop_dict = {}
        self.n_restart -= 1

    def _compute_netwon_step(self, X: np.ndarray, Y: np.ndarray, dual_vars: np.ndarray = None) -> Dict:
        out = self.hypervolume_derivatives.compute(X, Y)
        dHV, ddHV = out["HVdX"], out["HVdX2"]

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
                    np.concatenate([ddHV + ddH, dH.T], axis=1),
                    np.concatenate([dH, np.zeros((mup, mup))], axis=1),
                ],
            )
            newton_step = np.linalg.solve(dG, G)  # compute the Netwon's step
            # TODO: perhaps using Cholesky decomposition to speed up
            # newton_step = cho_solve(cho_factor(dG), G)
            return dict(step_X=newton_step[0:mud], step_dual=newton_step[mud:], H=H)
        else:
            try:
                newton_step = np.linalg.solve(ddHV, dHV.ravel())
                # newton_step = cho_solve(cho_factor(ddHV), dHV)
            except:
                breakpoint()
            # TODO: check if the `ddHV` is negtive definite
            # (w, _) = np.linalg.eigh(ddHV)
            return dict(step_X=newton_step)

    def step(self):
        # get unique points
        idx = np.unique(self.X, axis=0, return_index=True)[1]
        if len(idx) != self.mu:
            breakpoint()
        self.mu = len(idx)
        self.X, self.Y, self.dual_vars = self.X[idx, :], self.Y[idx, :], self.dual_vars[idx, :]
        # partition approximation set to anti-chains
        fronts = non_domin_sort(self.Y, only_front_indices=True)
        if self.h is not None:
            self.eq_cstr_value = np.zeros((self.mu, self.n_eq_cstr))

        # Newton-Raphson method for each front
        for _, idx in fronts.items():
            N = len(idx)
            X, Y = self.X[idx, :], self.Y[idx, :]
            lambdas = self.dual_vars[idx, :] if self.h is not None else None
            out = self._compute_netwon_step(X, Y, lambdas)
            self.X[idx, :] -= out["step_X"].reshape(N, -1)
            if self.h is not None:
                self.dual_vars[idx, :] -= out["step_dual"].reshape(N, -1)
                self.eq_cstr_value[idx, :] = out["H"]

        # evaluation
        self.Y = np.array([self.func(x) for x in self.X])
        self.iter_count += 1

    def log(self):
        HV = hypervolume(self.Y, self.ref)
        self.hist_Y += [self.Y.copy()]
        self.hist_X += [self.X.copy()]
        self.hist_HV += [HV]

        if self.verbose:
            self.logger.info(f"iteration {self.iter_count} ---")
            self.logger.info(f"X {self.X.ravel()}")
            self.logger.info(f"HV {HV}")

        if self.iter_count >= 1:
            self._delta_X = np.mean(np.sqrt(np.sum((self.hist_X[-1] - self.hist_X[-2]) ** 2, axis=1)))
            self._delta_Y = np.mean(np.sqrt(np.sum((self.hist_Y[-1] - self.hist_Y[-2]) ** 2, axis=1)))
            self._delta_HV = np.abs(self.hist_HV[-1] - self.hist_HV[-2])

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

        if self._delta_HV < self.HVtol:
            self.stop_dict["HVtol"] = self._delta_HV
            self.stop_dict["iter_count"] = self.iter_count

        if self._delta_X < self.xtol:
            self.stop_dict["xtol"] = self._delta_X
            self.stop_dict["iter_count"] = self.iter_count

        return bool(self.stop_dict)


# TODO: check if we need line search for the step-size
# try:
#     alphak, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(
#         f, fprime, xk, pk, gfk, old_fval, old_old_fval
#     )
# except _LineSearchError:
#     # Line search failed to find a better solution.
#     msg = "Warning: " + _status_message["pr_loss"]
#     return terminate(2, msg)
