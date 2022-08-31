import logging
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.linalg import block_diag, cholesky
from scipy.spatial.distance import cdist

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

        self._grad_norm = 0

    def _initialize(self, X0):
        if X0 is not None:
            X0 = np.asarray(X0)
            # assert np.all(X0 - self.lower_bounds >= 0)
            # assert np.all(X0 - self.upper_bounds <= 0)
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
        self._max_HV = np.product(self.ref)
        # initialize dual variables
        if self.h is not None:
            v = self.h(self.X[0, :])
            self.n_eq_cstr = 1 if isinstance(v, float) else len(v)
            # TODO: what is the best way to initialize lambdas
            self.dual_vars = np.ones((self.mu, self.n_eq_cstr)) / self.mu  # of size (mu, p)
            self.eq_cstr = np.atleast_2d([[self.h(_)] for _ in self.X])
            # to make the Hessian of Eq. constraints always a 3D tensor
            self._h_hessian = lambda x: self.h_hessian(x).reshape(self.n_eq_cstr, self.dim, -1)

    def _init_logging_var(self):
        """parameters for logging the history"""
        self.hist_Y: List[np.ndarray] = []
        self.hist_X: List[np.ndarray] = []
        self.hist_HV: List[float] = []
        self.hist_N_nondominated: List[int] = []
        self._delta_X: float = np.inf
        self._delta_Y: float = np.inf
        self._delta_HV: float = np.inf
        self._hist_inner = []
        self._hist_n_positive_eigen = []

        if self.h is not None:
            self.hist_dual_vars: List[np.ndarray] = []
            self.hist_eq_cstr: List[np.ndarray] = []
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
            self.log()
        return self.X, self.Y, self.stop_dict

    def _precondition_hessian(self, H: np.ndarray) -> np.ndarray:
        """Precondition the Hessian matrix to make sure it is negative definite

        Args:
            H (np.ndarray): the Hessian matrix

        Returns:
            np.ndarray: the preconditioned Hessian
        """
        # pre-condition the Hessian
        beta = 1e-6
        v = np.min(np.diag(-H))
        tau = 0 if v > 0 else -v + beta
        I = np.eye(H.shape[0])
        for _ in range(35):
            try:
                _ = cholesky(-H + tau * I, lower=True)
                break
            except:
                tau = max(1.3 * tau, beta)
        else:
            self.logger.warn("Pre-conditioning the HV Hessian failed")
        return H - tau * I

    def _compute_G(self, X: np.ndarray, dual_vars: np.ndarray) -> np.ndarray:
        N = len(X)
        # mud = int(N * self.dim)
        # dHV = np.zeros(mud)
        # Y = np.array([self.func(x) for x in X])
        # fronts = non_domin_sort(Y, only_front_indices=True)
        # for i, front in fronts.items():
        out = self.hypervolume_derivatives.compute(X)
        # normalize the HV value to avoid large gradient values
        # idx = np.r_[tuple([slice(i * self.dim, (i + 1) * self.dim) for i in front])]
        # dHV[idx] = out["HVdX"]
        dHV = out["HVdX"].ravel()  # / self._max_HV

        dH = block_diag(*[self.h_jac(x) for x in X])
        eq_cstr = np.array([self.h(_) for _ in X]).reshape(X.shape[0], -1)
        G = np.concatenate([dHV + dual_vars.ravel() @ dH, eq_cstr.ravel()])
        G = np.c_[G[: N * self.dim].reshape(N, -1), G[N * self.dim :].reshape(N, -1)]
        return G

    def _compute_netwon_step(self, X: np.ndarray, Y: np.ndarray, dual_vars: np.ndarray = None) -> Dict:
        N = X.shape[0]
        mud = int(N * self.dim)
        # ddHV = np.zeros((mud, mud))
        # dHV = np.zeros(mud)
        out = self.hypervolume_derivatives.compute(X, Y)
        # normalize the HV value to avoid large gradient values
        # dHV, ddHV = out["HVdX"] / self._max_HV, out["HVdX2"] / self._max_HV
        # idx = np.r_[tuple([slice(i * self.dim, (i + 1) * self.dim) for i in front])]
        # ddHV[np.ix_(idx, idx)] = out["HVdX2"]
        # dHV[idx] = out["HVdX"]
        dHV, ddHV = out["HVdX"], out["HVdX2"]
        # ddHV = self._precondition_hessian(ddHV)
        H, g = ddHV, dHV.ravel()
        w = np.linalg.eigh(H)[0]
        # if np.any(w < 0):
        #     from autograd import hessian

        #     def _hypervolumeX(x):
        #         return hypervolume(np.array([self.func(x_) for x_ in x.reshape(-1, 3)]), self.ref)

        #     HVX_Hessian = hessian(_hypervolumeX)
        #     H_ = HVX_Hessian(X.ravel())
        #     w_ = np.linalg.eigh(H_)[0]
        #     assert np.all(np.isclose(H, H_))

        if self.h is not None:  # with equality constraints
            N = X.shape[0]
            mud = int(N * self.dim)
            mup = int(N * self.n_eq_cstr)
            eq_cstr = np.array([self.h(_) for _ in X]).reshape(N, -1)  # (mu, p)
            dH_ = np.array([self.h_jac(x) for x in X])
            dH = block_diag(*dH_)  # (mu * p, dim)
            # NOTE: the constraint Hessian can be dropped without decreasing the performance
            ddH = block_diag(
                *[np.einsum("ijk,i->jk", self._h_hessian(x), dual_vars[i]) for i, x in enumerate(X)]
            )  # (mu * dim, mu * dim)
            # M = self._precondition_hessian(ddHV + ddH)
            g = np.concatenate([dHV.ravel() + dual_vars.ravel() @ dH, eq_cstr.ravel()])
            H = np.concatenate(
                [
                    np.concatenate([ddHV + ddH, dH.T], axis=1),
                    np.concatenate([dH, np.zeros((mup, mup))], axis=1),
                ],
            )
        try:
            newton_step = -1 * np.linalg.solve(H, g)
        except:
            w, V = np.linalg.eigh(H)
            w[np.isclose(w, 0)] = 1e-6
            D = np.diag(1 / w)
            newton_step = -1 * V @ D @ V.T @ g

        if self.h is not None:
            return dict(
                step_X=newton_step[0:mud],
                step_dual=newton_step[mud:],
                # step_X=step_X,
                # step_dual=step_dual,
                grad=dHV.ravel(),
                HVdX=out["HVdX"],
                HVdY=out["HVdY"],
                HVdY2=out["HVdY2"],
                G=g,
                w=w,
                # eq_cstr=eq_cstr,
                # dH=dH_,
                # H=H,
            )
        else:
            return dict(step_X=newton_step, grad=dHV.ravel(), w=w)

    def step(self):
        def _HV(x):
            return self.hypervolume_derivatives.HV(x)  # / self._max_HV

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
        # idx = list(set(range(self.mu)) - set(drop_idx_X))
        self.mu = len(idx)
        self.X = self.X[idx, :]
        self.Y = self.Y[idx, :]

        if self.h is not None:
            self.dual_vars = self.dual_vars[idx, :]
            self.step_dual = np.zeros((self.mu, self.n_eq_cstr))
            self.G = np.zeros((self.mu, self.n_eq_cstr + self.dim))
            self.eq_cstr = self.eq_cstr[idx, :]
            self.dH = np.zeros((self.mu, self.n_eq_cstr, self.dim))

        self.step_X = np.zeros((self.mu, self.dim))
        self.grad = np.zeros((self.mu, self.dim))
        self.step_size = np.zeros(self.mu)

        # partition approximation set to anti-chains
        fronts = non_domin_sort(self.Y, only_front_indices=True)
        self._N_nondominated = len(fronts[0])
        self._nondominated_idx = fronts[0]

        for i, front in fronts.items():
            N = len(front)
            out = self._compute_netwon_step(
                X=self.X[front],
                Y=self.Y[front],
                dual_vars=self.dual_vars[front] if self.h is not None else None,
            )
            self.step_X[front, :] = out["step_X"].reshape(N, -1)
            self.grad[front, :] = out["grad"].reshape(N, -1)

            if self.h is not None:
                self.step_dual[front, :] = out["step_dual"].reshape(N, -1)
                self.G[front, :] = np.c_[
                    out["G"][: N * self.dim].reshape(N, -1), out["G"][N * self.dim :].reshape(N, -1)
                ]
                # self.eq_cstr = out["eq_cstr"]
                # self.dH = out["dH"].reshape(N, self.n_eq_cstr, -1)

        # backtracking line search with Armijo's condition for each point
        c = 1e-5
        for i, front in fronts.items():
            N = len(front)
            X = self.X[front]
            step_X = self.step_X[front]
            grad = self.grad[front]
            if i == 0:
                self._inner = np.inner(grad.ravel(), step_X.ravel()) / (
                    np.linalg.norm(grad.ravel()) * np.linalg.norm(step_X.ravel())
                )
                self._n_positive_eigen = np.sum(out["w"] > 0)
            # if np.inner(grad.ravel(), step_X.ravel()) < 0:
            # self.logger.warn("Newton direction is not ascending")
            # step_X = grad.copy()

            if self.h is not None:
                G = self.G[front]
                step_dual = self.step_dual[front]
                dual_vars = self.dual_vars[front]

            normal_vectors = np.c_[np.eye(self.dim * N), -1 * np.eye(self.dim * N)]
            # calculate the maximal step-size
            dist = np.r_[
                np.abs(X.ravel() - np.tile(self.lower_bounds, N)),
                np.abs(np.tile(self.upper_bounds, N) - X.ravel()),
            ]
            v = step_X.ravel() @ normal_vectors
            alpha0 = min(1, 0.25 * np.min(dist[v < 0] / np.abs(v[v < 0])))
            alpha = alpha0
            for _ in range(10):
                X_ = X + alpha * step_X
                if self.h is None:
                    impr = _HV(X_) - _HV(X)
                    cond = impr >= c * alpha * np.inner(grad.ravel(), step_X.ravel())
                else:
                    dual_vars_ = dual_vars + alpha * step_dual
                    G_ = self._compute_G(X_, dual_vars_)
                    # cond = np.sum(G_[self._nondominated_idx] ** 2) <= (1 - 2 * c) * np.sum(
                    #     self.G[self._nondominated_idx].ravel() ** 2
                    # )
                    cond = np.sum(G_**2) <= (1 - 2 * c) * np.sum(G.ravel() ** 2)
                if cond:
                    break
                else:
                    if 11 < 2:
                        phi0 = np.sum(G**2) / 2
                        phi1 = np.sum(G_**2) / 2
                        phi0prime = -np.sum(G**2)
                        tau = -phi0prime / (phi1 - phi0 - phi0prime) / 2
                        alpha *= tau
                    if 1 < 2:
                        alpha *= 0.5
            else:
                self.logger.info("Armijo's backtracking line search failed")

            self.step_size[front] = alpha

        self.X = self.X + self.step_size.reshape(-1, 1) * self.step_X
        if self.h is not None:
            self.dual_vars = self.dual_vars + self.step_size.reshape(-1, 1) * self.step_dual

        # evaluation
        self.Y = np.array([self.func(x) for x in self.X])
        self.iter_count += 1

    def log(self):
        HV = hypervolume(self.Y, self.ref)
        self.hist_Y += [self.Y.copy()]
        self.hist_X += [self.X.copy()]
        self.hist_HV += [HV]
        self.hist_N_nondominated += [self._N_nondominated]
        self._hist_inner += [self._inner]
        self._hist_n_positive_eigen += [self._n_positive_eigen]

        if self.verbose:
            self.logger.info(f"iteration {self.iter_count} ---")
            # self.logger.info(f"X: {self.X.ravel()}")
            self.logger.info(f"HV: {HV}")
            self.logger.info(f"step size: {self.step_size}")

        if self.iter_count >= 1:
            try:
                self._delta_X = np.mean(np.sqrt(np.sum((self.hist_X[-1] - self.hist_X[-2]) ** 2, axis=1)))
                self._delta_Y = np.mean(np.sqrt(np.sum((self.hist_Y[-1] - self.hist_Y[-2]) ** 2, axis=1)))
                self._delta_HV = np.abs(self.hist_HV[-1] - self.hist_HV[-2])
            except:
                pass

        if self.h is not None:
            # self.logger.info(f"dual variables: {self.dual_vars.ravel()}")
            # self.logger.info(f"Equality constraints: {self.eq_cstr.ravel()}")
            self.hist_dual_vars += [self.dual_vars]
            self.hist_eq_cstr += [self.eq_cstr]
            self.hist_G_norm += [np.linalg.norm(self.G[self._nondominated_idx].ravel())]
            self.logger.info(f"G norm: {np.linalg.norm(self.G[self._nondominated_idx].ravel())}")
            # self.hist_G_norm += [np.linalg.norm(self.G.ravel())]
            # self.logger.info(f"G norm: {np.linalg.norm(self.G.ravel())}")

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
