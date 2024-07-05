import logging
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.linalg import block_diag, cholesky, solve
from scipy.spatial.distance import cdist

from .base import State
from .delta_p import GenerationalDistance, InvertedGenerationalDistance
from .hypervolume_derivatives import HypervolumeDerivatives
from .utils import (compute_chim, get_logger, non_domin_sort,
                    precondition_hessian, set_bounds)

__authors__ = ["Hao Wang"]

np.seterr(divide="ignore", invalid="ignore")


def Nd_vector_to_matrix(
    x: np.ndarray, N: int, dim: int, dim_primal: int, active_indices: np.ndarray
) -> np.ndarray:
    if dim == dim_primal:  # the unconstrained case
        return x.reshape(N, -1)
    X = np.zeros((N, dim))  # Netwon steps
    D = int(N * dim_primal)
    a, b = x[:D].reshape(N, -1), x[D:]
    idx = np.r_[0, np.cumsum([sum(k) - dim_primal for k in active_indices])]
    b = [b[idx[i] : idx[i + 1]] for i in range(len(idx) - 1)]
    for i, idx in enumerate(active_indices):
        X[i, idx] = np.r_[a[i], b[i]]
    return X


class HVN:
    """Hypervolume Newton method

    Newton-Raphson method to maximize the hypervolume indicator, subject to equality constraints
    """

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        func: callable,
        jac: callable,
        hessian: callable,
        ref: Union[List[float], np.ndarray],
        xl: Union[List[float], np.ndarray],
        xu: Union[List[float], np.ndarray],
        N: int = 5,
        h: Callable = None,
        h_jac: callable = None,
        h_hessian: callable = None,
        g: Callable = None,
        g_jac: Callable = None,
        g_hessian: callable = None,
        X0: np.ndarray = None,
        max_iters: Union[int, str] = np.inf,
        xtol: float = 1e-3,
        verbose: bool = True,
        problem_name: str = None,
        metrics: Dict[str, Callable] = dict(),
        preconditioning: bool = False,
        **kwargs,
    ):
        self.dim_p: int = n_var  # the number of primal variables
        self.n_obj: int = n_obj  # the number of objectives
        self.N: int = N  # the population size
        self.xl: np.ndarray = xl
        self.xu: np.ndarray = xu
        self.ref: np.ndarray = ref
        self._check_constraints(h, g)
        self.state = State(
            self.dim_p,
            self.n_eq,
            self.n_ieq,
            func,
            jac,
            h=h,
            h_jac=h_jac,
            h_hess=h_hessian,
            g=g,
            g_jac=g_jac,
            g_hess=g_hessian,
        )
        self.indicator = HypervolumeDerivatives(self.dim_p, self.n_obj, ref, func, jac, hessian)
        self._initialize(X0)
        self._set_logging(verbose)
        self.xtol: float = xtol
        self.max_iters: int = self.N * 10 if max_iters is None else max_iters
        self.stop_dict: Dict[str, float] = {}
        self.metrics = metrics
        self.preconditioning: bool = preconditioning
        self.verbose: bool = verbose
        self.eps = 1e-3 * np.max(self.xu - self.xl)
        self.problem_name = problem_name

    def _check_constraints(self, h: Callable, g: Callable):
        # initialize dual variables
        self.n_eq, self.n_ieq = 0, 0
        self._constrained = h is not None or g is not None
        x = np.random.rand(self.dim_p) * (self.xu - self.xl) + self.xl
        if h is not None:
            v = h(x)
            self.n_eq = 1 if isinstance(v, (int, float)) else len(v)
        if g is not None:
            v = g(x)
            self.n_ieq = 1 if isinstance(v, (int, float)) else len(v)
        self.dim_d = self.n_eq + self.n_ieq
        self.dim = self.dim_p + self.dim_d

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
            X0 = np.random.rand(self.N, self.dim_p) * (self.xu - self.xl) + self.xl  # (mu, dim_primal)
        # initialize the state variables
        self.state.update(np.c_[X0, np.zeros((self.N, self.dim_d)) / self.N])  # (mu, dim)
        self.iter_count: int = 0
        self._max_HV = np.product(self.ref)  # TODO: this should be moved to `HV` class

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
        self._xl = set_bounds(lb, self.dim_p)

    @property
    def xu(self):
        return self._xu

    @xu.setter
    def xu(self, ub):
        self._xu = set_bounds(ub, self.dim_p)

    def run(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        while not self.terminate():
            self.newton_iteration()
            self.log()
        return self.state.primal, self.state.Y, self.stop_dict

    def _compute_R(
        self, state: State, grad: np.ndarray = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """compute the root-finding problem R
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
                (R, H, active_indices) -> the rooting-finding problem,
                the Jacobian of the equality constraints, and
                the indices of the active primal-dual variables
        """
        if grad is None:
            grad = self.indicator.compute_derivatives(state.primal, state.Y, False, state.J)
        # the unconstrained case
        R, dH, active_indices = grad, None, np.array([[True] * state.n_var] * state.N)
        if self._constrained:
            R = np.zeros((state.N, self.dim))  # the root-finding problem
            func = lambda g, dual, h: g + np.einsum("j,jk->k", dual, h)
            cstr_value, idx, dH = state.cstr_value, state.active_indices, state.cstr_grad
            dH = [dH[i, idx, :] for i, idx in enumerate(idx)]  # only take Jacobian of the active constraints
            active_indices = np.c_[active_indices, idx]
            for i, k in enumerate(active_indices):
                R[i, k] = np.r_[func(grad[i], state.dual[i, idx[i]], dH[i]), cstr_value[i, idx[i]]]
        return R, dH, active_indices

    def _compute_netwon_step(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        grad, Hessian = self.indicator.compute_derivatives(X=state.primal, Y=state.Y, YdX=state.J)
        R, H, active_indices = self._compute_R(state, grad=grad)
        # sometimes the Hessian is not NSD
        if self.preconditioning:
            Hessian = self._precondition_hessian(Hessian)
        DR, idx = Hessian, active_indices[:, self.dim_p :]
        if self._constrained:
            H = block_diag(*H)  # (N * p, N * dim), `p` is the number of active constraints
            B = state.cstr_hess
            M = block_diag(*[np.einsum("i...,i", B[i, k], state.dual[i, k]) for i, k in enumerate(idx)])
            Z = np.zeros((len(H), len(H)))
            DR = np.r_[np.c_[DR + M, H.T], np.c_[H, Z]]
        # the vector-format of R
        R_ = np.r_[R[:, : self.dim_p].reshape(-1, 1), R[:, self.dim_p :][idx].reshape(-1, 1)]
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                newton_step_ = -1 * solve(DR, R_)
            except:  # if DR is singular, then use the pseudoinverse
                newton_step_ = -1 * np.linalg.lstsq(DR, R_, rcond=None)[0]
        # convert the vector-format of the newton step to matrix format
        newton_step = Nd_vector_to_matrix(newton_step_.ravel(), state.N, self.dim, self.dim_p, active_indices)
        return newton_step, R

    def _compute_indicator_value(self):
        self.curr_indicator_value = self.indicator.compute(Y=self.state.Y)

    def newton_iteration(self):
        # check for anomalies in `X` and `Y`
        self._check_population()
        # first compute the current indicator value
        self._compute_indicator_value()
        self._nondominated_idx = non_domin_sort(self.state.Y, only_front_indices=True)[0]
        # partition the approximation set to by feasibility
        feasible_mask = self.state.is_feasible()
        feasible_idx = np.nonzero(feasible_mask)[0]
        infeasible_idx = np.nonzero(~feasible_mask)[0]
        partitions = {0: np.array(range(self.N))}
        if len(feasible_idx) > 0:
            # non-dominatd sorting of the feasible points
            partitions = non_domin_sort(self.state.Y[feasible_idx], only_front_indices=True)
            partitions = {k: feasible_idx[v] for k, v in partitions.items()}
            # add all infeasible points to the first dominance layer
            partitions.update({0: np.sort(np.r_[partitions[0], infeasible_idx])})

        # compute the Newton direction for each partition
        self.step = np.zeros((self.N, self.dim))
        self.step_size = np.ones(self.N)
        self.R = np.zeros((self.N, self.dim))
        for i, idx in partitions.items():
            # compute Newton step
            newton_step, R = self._compute_netwon_step(self.state[idx])
            self.step[idx, :] = newton_step
            self.R[idx, :] = R
            # backtracking line search with Armijo's condition for each layer
            idx_ = list(set(idx) - set(infeasible_idx)) if i == 0 else idx
            self.step_size[idx_] = self._line_search(self.state[idx_], self.step[idx_], R=self.R[idx_])
        # TODO: unify `_line_search` and `_line_search_dominated`
        # for k in infeasible_idx:  # for dominated and infeasible points
        # self.step_size[k] = self._line_search(self.state[infeasible_idx], self.step[infeasible_idx], R=self.R[infeasible_idx])
        # TODO: this part should be tested
        self.step_size[infeasible_idx] = self._line_search(
            self.state[infeasible_idx], self.step[infeasible_idx], R=self.R[infeasible_idx]
        )
        # Newton iteration and evaluation
        self.state.update(self.state.X + self.step_size.reshape(-1, 1) * self.step)

    def log(self):
        self.iter_count += 1
        self.history_X += [self.state.primal.copy()]
        self.history_Y += [self.state.Y.copy()]
        self.history_indicator_value += [self.curr_indicator_value]
        self.history_R_norm += [np.median(np.linalg.norm(self.R, axis=1))]
        for name, func in self.metrics.items():  # compute the performance metrics
            self.history_metrics[name].append(func.compute(Y=self.state.Y))
        if self.verbose:
            self.logger.info(f"iteration {self.iter_count} ---")
            self.logger.info(f"{self.indicator.__class__.__name__}: {self.curr_indicator_value}")
            self.logger.info(f"step size: {self.step_size.ravel()}")
            self.logger.info(f"R norm: {self.history_R_norm[-1]}")
            self.logger.info(f"#non-dominated: {len(self._nondominated_idx)}")

    def terminate(self) -> bool:
        if self.iter_count >= self.max_iters:
            self.stop_dict["iter_count"] = self.iter_count

        return bool(self.stop_dict)

    def _precondition_hessian(self, H: np.ndarray) -> np.ndarray:
        """Precondition the Hessian matrix to make sure it is negative definite

        Args:
            H (np.ndarray): the Hessian matrix

        Returns:
            np.ndarray: the preconditioned Hessian
        """
        # TODO: remove this function
        # pre-condition the Hessian
        beta = 1e-6
        v = np.min(np.diag(-H))
        tau = 0 if v > 0 else -v + beta
        I = np.eye(H.shape[0])
        for _ in range(35):
            try:
                cholesky(-H + tau * I, lower=True)
                break
            except:
                # NOTE: the multiplier is not working for Eq1IDTLZ3.. Otherwise, it takes 1.5
                tau = max(1.5 * tau, beta)
        else:
            self.logger.warn("Pre-conditioning the HV Hessian failed")
        return H - tau * I

    def _line_search(
        self, state: State, step: np.ndarray, R: np.ndarray, max_step_size: float = None
    ) -> float:
        """backtracking line search with Armijo's condition"""
        c1 = 1e-4
        if np.all(np.isclose(step, 0)):
            return 1

        def phi_func(alpha):
            state_ = deepcopy(state)
            state_.update(state.X + alpha * step)
            R = self._compute_R(state_)[0]
            return np.linalg.norm(R)

        step_size = 1 if max_step_size is None else max_step_size
        phi = [np.linalg.norm(R)]
        s = [0, step_size]
        for _ in range(6):
            phi.append(phi_func(s[-1]))
            # Armijo–Goldstein condition
            # when R norm is close to machine precision, it makes no sense to perform the line search
            success = phi[-1] <= (1 - c1 * s[-1]) * phi[0] or np.isclose(phi[0], np.finfo(float).eps)
            if success:
                break
            else:
                if 11 < 2:
                    # cubic interpolation to compute the next step length
                    d1 = -phi[-2] - phi[-1] - 3 * (phi[-2] - phi[-1]) / (s[-2] - s[-1])
                    d2 = np.sign(s[-1] - s[-2]) * np.sqrt(d1**2 - phi[-2] * phi[-1])
                    s_ = s[-1] - (s[-1] - s[-2]) * (-phi[-1] + d2 - d1) / (-phi[-1] + phi[-2] + 2 * d2)
                    s_ = s[-1] * 0.5 if np.isnan(s_) else np.clip(s_, 0.4 * s[-1], 0.6 * s[-1])
                    s.append(s_)
                else:
                    s.append(s[-1] * 0.5)
        else:
            self.logger.warn("backtracking line search failed")
        step_size = s[-1]
        return step_size

    def _line_search_dominated(self, X: np.ndarray, step: np.ndarray) -> float:
        """backtracking line search with Armijo's condition"""
        # TODO: ad-hoc! to solve this in the further using a high precision numerical library
        # NOTE: when the step length is close to numpy's numerical resolution, it makes no sense to perform
        # the step-size control
        if np.any(np.isclose(np.median(step), np.finfo(np.double).resolution)):
            return 1

        c = 1e-4
        N = len(X)
        step = step[:, : self.dim_p]
        primal_vars = self._get_primal_dual(X)[0]
        normal_vectors = np.c_[np.eye(self.dim_p * N), -1 * np.eye(self.dim_p * N)]
        # calculate the maximal step-size
        dist = np.r_[
            np.abs(primal_vars.ravel() - np.tile(self.xl, N)),
            np.abs(np.tile(self.xu, N) - primal_vars.ravel()),
        ]
        v = step.ravel() @ normal_vectors
        alpha = min(1, 0.25 * np.min(dist[v < 0] / np.abs(v[v < 0])))

        h_ = np.array([self.h(x) for x in primal_vars])
        eq_cstr = h_**2 / 2
        G = h_ * np.array([self.h_jac(x) for x in primal_vars])
        for _ in range(6):
            X_ = primal_vars + alpha * step
            eq_cstr_ = np.array([self.h(x) for x in X_]) ** 2 / 2
            dec = np.inner(G.ravel(), step.ravel())
            cond = eq_cstr_ - eq_cstr <= c * alpha * dec
            if cond:
                break
            else:
                alpha *= 0.5
        else:
            self.logger.warn("Armijo's backtracking line search failed")
        return alpha

    def _check_population(self):
        # get unique points: if some points converge to the same location
        primal_vars = self.state.primal
        D = cdist(primal_vars, primal_vars)
        drop_idx_X = set([])
        for i in range(self.N):
            if i not in drop_idx_X:
                drop_idx_X |= set(np.nonzero(np.isclose(D[i, :], 0, rtol=self.eps))[0]) - set([i])

        # get rid of weakly-dominated points
        drop_idx_Y = set([])
        # TODO: Ad-hoc solution! check if this is still needed
        # if self.problem_name is not None and self.problem_name not in ("Eq1DTLZ4", "Eq1IDTLZ4"):
        #     for i in range(self.N):
        #         if i not in drop_idx_Y:
        #             drop_idx_Y |= set(np.nonzero(np.any(np.isclose(self.Y[i, :], self.Y), axis=1))[0]) - set(
        #                 [i]
        #             )
        idx = list(set(range(self.N)) - (drop_idx_X | drop_idx_Y))
        self.state = self.state[idx]
        self.N = self.state.N


class DpN:
    """Delta_p Newton method with constraint handling

    Newton-Raphson method to minimize the Delta_p indicator, subject to equality/inequality constraints.
    The equalities are handled locally with the KKT condition.
    The inequalities are converted to equalities via the active set method.
    """

    def __init__(
        self,
        dim: int,
        n_obj: int,
        func: callable,
        jac: callable,
        hessian: callable,
        ref: Union[List[float], np.ndarray],
        xl: Union[List[float], np.ndarray],
        xu: Union[List[float], np.ndarray],
        type: str = "deltap",
        N: int = 5,
        h: Callable = None,
        h_jac: callable = None,
        g: Callable = None,
        g_jac: Callable = None,
        x0: np.ndarray = None,
        max_iters: Union[int, str] = np.inf,
        xtol: float = 0,
        verbose: bool = True,
        pareto_front: Union[List[float], np.ndarray] = None,
        eta=None,
        Y_label=None,
    ):
        """
        Args:
            dim (int): dimensionality of the search space.
            n_objective (int): number of objectives
            func (callable): the objective function to be minimized.
            jac (callable): the Jacobian of objectives, should return a matrix of size (n_objective, dim)
            hessian (callable): the Hessian of objectives,
                should return a Tensor of size (n_objective, dim, dim)
            ref (Union[List[float], np.ndarray]): the reference set, of shape (n_point, n_objective)
            lower_bounds (Union[List[float], np.ndarray], optional): the lower bound of search variables.
                When it is not a `float`, it must have shape (dim, ).
            upper_bounds (Union[List[float], np.ndarray], optional): The upper bound of search variables.
                When it is not a `float`, it must have must have shape (dim, ).
            N (int, optional): the approximation set size. Defaults to 5.
            h (Callable, optional): the equality constraint function, should return a vector of shape
                (n_equality, ). Defaults to None.
            h_jac (callable, optional): the Jacobian of equality constraint function,
                should return a matrix of (n_equality, dim). Defaults to None.
            g (Callable, optional): the inequality constraint function, should return a vector of shape
                (n_inequality, ). Defaults to None.
            g_jac (callable, optional): the Jacobian of the inequality constraint function,
                should return a matrix of (n_inequality, dim). Defaults to None.
            h_hessian (callable, optional): the Jacobian of constraint function.
                should return a matrix of (n_inequality, dim, dim). Defaults to None.
            x0 (np.ndarray, optional): the initial approximation set, of shape (mu, dim). Defaults to None.
            max_iters (Union[int, str], optional): maximal iterations of the algorithm. Defaults to np.inf.
            xtol (float, optional): absolute distance in the approximation set between consecutive iterations
                that is used to determine convergence. Defaults to 1e-3.
            verbose (bool, optional): verbosity of the output. Defaults to True.
        """
        assert type in ["gd", "igd", "deltap"]
        self.type = type
        self.dim_p = dim
        self.n_obj = n_obj
        self.N = N
        self.xl = xl
        self.xu = xu
        self._check_constraints(h, g)
        self.state = State(self.dim_p, self.n_eq, self.n_ieq, func, jac, h=h, h_jac=h_jac, g=g, g_jac=g_jac)
        self._initialize(x0)
        self._set_indicator(ref, func, jac, hessian)
        self._set_logging(verbose)
        # parameters controlling stop criteria
        self.xtol = xtol
        self.max_iters: int = self.N * 10 if max_iters is None else max_iters
        self.stop_dict: Dict = {}
        # TODO: implement those as callbacks
        self._pareto_front = pareto_front
        self._perf_gd = GenerationalDistance(ref=pareto_front)
        self._perf_igd = InvertedGenerationalDistance(ref=pareto_front, cluster_matching=False)
        # TODO: fix those ad-hoc variable for shifting and clustering of `Y`
        self._eta = eta
        self.Y_label = Y_label
        if self.Y_label is None:
            self.Y_label = np.array([0] * self.N, dtype=int)
            self.n_cluster = 1
            self.Y_idx = [list(range(self.N))]
        else:
            self.n_cluster = len(np.unique(self.Y_label))
            self.Y_idx = [np.nonzero(self.Y_label == i)[0] for i in range(self.n_cluster)]

    def _check_constraints(self, h: Callable, g: Callable):
        # initialize dual variables
        self.n_eq, self.n_ieq = 0, 0
        self._constrained = h is not None or g is not None
        x = np.random.rand(self.dim_p) * (self.xu - self.xl) + self.xl
        if h is not None:
            v = h(x)
            self.n_eq = 1 if isinstance(v, (int, float)) else len(v)
        if g is not None:
            v = g(x)
            self.n_ieq = 1 if isinstance(v, (int, float)) else len(v)
        self.dim_d = self.n_eq + self.n_ieq
        self.dim = self.dim_p + self.dim_d

    def _initialize(self, X0: np.ndarray):
        if X0 is not None:
            X0 = np.asarray(X0)
            X0 = np.clip(X0, self.xl, self.xu)
            # NOTE: ad-hoc solution for CF2 and IDTLZ1 since the Jacobian on the box boundary is not defined
            # on the decision boundary or the local Hessian is ill-conditioned.
            X0 = np.clip(X0 - self.xl, 1e-2, 1) + self.xl
            X0 = np.clip(X0 - self.xu, -1, -1e-2) + self.xu
            self.N = len(X0)
        else:
            # sample `x` u.a.r. in `[lb, ub]`
            assert self.N is not None
            assert all(~np.isinf(self.xl)) & all(~np.isinf(self.xu))
            X0 = np.random.rand(self.N, self.dim_p) * (self.xu - self.xl) + self.xl  # (mu, dim_primal)
        # initialize the state variables
        self.state.update(np.c_[X0, np.zeros((self.N, self.dim_d)) / self.N])  # (mu, dim)
        self.iter_count: int = 0

    def _set_indicator(
        self, ref: Union[np.ndarray, Dict[int, np.ndarray]], func: Callable, jac: Callable, hessian: Callable
    ):
        self._gd = GenerationalDistance(ref, func, jac, hessian)
        self._igd = InvertedGenerationalDistance(ref, func, jac, hessian, cluster_matching=True)

    def _set_logging(self, verbose):
        """parameters for logging the history"""
        self.verbose: bool = verbose
        self.GD_value: float = None
        self.IGD_value: float = None
        self.hist_Y: List[np.ndarray] = []
        self.hist_X: List[np.ndarray] = []
        self.hist_GD: List[float] = []
        self.hist_IGD: List[float] = []
        self._delta_X: float = np.inf
        self._delta_Y: float = np.inf
        self._delta_GD: float = np.inf
        self._delta_IGD: float = np.inf
        self.hist_R_norm: List[float] = []
        self.history_medoids: Dict[int, List] = dict()
        self.logger: logging.Logger = get_logger(
            logger_id=f"{self.__class__.__name__}",
            console=self.verbose,
        )

    @property
    def xl(self):
        return self._xl

    @xl.setter
    def xl(self, lb: np.ndarray):
        self._xl = set_bounds(lb, self.dim_p)

    @property
    def xu(self):
        return self._xu

    @xu.setter
    def xu(self, ub: np.ndarray):
        self._xu = set_bounds(ub, self.dim_p)

    @property
    def active_indicator(
        self,
    ) -> Union[GenerationalDistance, InvertedGenerationalDistance]:
        """return the incumbent performance indicator."""
        if self.type == "deltap":
            indicator = self._gd if self.GD_value > self.IGD_value else self._igd
        else:
            indicator = self._gd if self.type == "gd" else self._igd
        return indicator

    def run(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        while not self.terminate():
            self.newton_iteration()
            self.log()
        return self.state.primal, self.state.Y, self.stop_dict

    def newton_iteration(self):
        # compute the initial indicator values. The first clustering and matching is executed here.
        self._compute_indicator_value(self.state.Y)
        # shift the reference set if needed
        self._shift_reference_set()
        # compute the Newton step
        self.step, self.R = self._compute_netwon_step()
        # prevent the decision points from moving out of the decision space. Needed for CF7 with NSGA-III
        self.step, max_step_size = self._handle_box_constraint(self.step)
        # backtracking line search for the step size
        self.step_size = self._backtracking_line_search(self.step, self.R, max_step_size)
        # Newton iteration and evaluation
        self.state.update(self.state.X + self.step_size * self.step)

    def log(self):
        self.iter_count += 1
        self.hist_Y += [self.state.Y.copy()]
        self.hist_X += [self.state.primal.copy()]
        gd_value = self._perf_gd.compute(Y=self.state.Y)
        igd_value = self._perf_igd.compute(Y=self.state.Y)
        self.hist_GD += [gd_value]
        self.hist_IGD += [igd_value]
        self.hist_R_norm += [np.median(np.linalg.norm(self.R, axis=1))]

        if self.iter_count >= 2:
            self._delta_X = np.mean(np.sqrt(np.sum((self.hist_X[-1] - self.hist_X[-2]) ** 2, axis=1)))
            self._delta_Y = np.mean(np.sqrt(np.sum((self.hist_Y[-1] - self.hist_Y[-2]) ** 2, axis=1)))
            self._delta_GD = np.abs(self.hist_GD[-1] - self.hist_GD[-2])
            self._delta_IGD = np.abs(self.hist_IGD[-1] - self.hist_IGD[-2])

        if self.verbose:
            self.logger.info(f"iteration {self.iter_count} ---")
            self.logger.info(f"GD/IGD: {self.GD_value, igd_value}")
            # self.logger.info(f"step size: {self.step_size.ravel()}")
            self.logger.info(f"R norm: {self.hist_R_norm[-1]}")

    def terminate(self) -> bool:
        if self.iter_count >= self.max_iters:
            self.stop_dict["iter_count"] = self.iter_count

        if self._delta_X < self.xtol:
            self.stop_dict["xtol"] = self._delta_X
            self.stop_dict["iter_count"] = self.iter_count

        return bool(self.stop_dict)

    def _compute_indicator_value(self, Y: np.ndarray):
        self.GD_value = self._gd.compute(Y=Y)
        self.IGD_value = self._igd.compute(Y=Y, Y_label=self.Y_label)

    def _compute_R(
        self, state: State, grad: np.ndarray = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """compute the root-finding problem R
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
                (R, H, idx) -> the rooting-finding problem,
                the Jacobian of the equality constraints, and
                the indices that are active among primal-dual variables
        """
        primal_vars, dual_vars = state.primal, state.dual
        if grad is None:
            grad = self.active_indicator.compute_derivatives(
                X=primal_vars, Y=state.Y, Y_label=self.Y_label, compute_hessian=False, Jacobian=state.J
            )
        R = grad  # the unconstrained case
        dH, idx = None, None
        if self._constrained:
            func = lambda g, dual, h: g + np.einsum("j,jk->k", dual, h)
            v, idx, dH = state.cstr_value, state.active_indices, state.cstr_grad
            dH = [dH[i][idx, :] for i, idx in enumerate(idx)]
            R = [np.r_[func(grad[i], dual_vars[i, k], dH[i]), v[i, k]] for i, k in enumerate(idx)]
        return R, dH, idx

    def _compute_netwon_step(self) -> Tuple[np.ndarray, np.ndarray]:
        primal_vars = self.state.primal
        newton_step = np.zeros((self.N, self.dim))  # Netwon steps
        R = np.zeros((self.N, self.dim))  # the root-finding problem
        # gradient and Hessian of the incumbent indicator
        grad, Hessian = self.active_indicator.compute_derivatives(
            X=primal_vars,
            Y=self.state.Y,
            Jacobian=self.state.J,
            Y_label=self.Y_label,
        )
        # the root-finding problem and the gradient of the active constraints
        R_list, dH, active_indices = self._compute_R(self.state, grad=grad)
        idx = np.array([[True] * self.dim_p] * self.N)
        if active_indices is not None:
            idx = np.c_[idx, active_indices]
        # compute the Newton step for each approximation point - lower computation costs
        for r in range(self.N):
            c = idx[r]
            dh = np.array([]) if dH is None else dH[r]
            Z = np.zeros((len(dh), len(dh)))
            # pre-condition indicator's Hessian if needed, e.g., on ZDT6, CF1, CF7
            Hessian[r] = precondition_hessian(Hessian[r])
            # derivative of the root-finding problem
            DR = np.r_[np.c_[Hessian[r], dh.T], np.c_[dh, Z]] if self._constrained else Hessian[r]
            R[r, c] = R_list[r]
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    newton_step[r, c] = -1 * solve(DR, R_list[r].reshape(-1, 1)).ravel()
                except Exception:
                    # if DR is singular, then use the pseudoinverse.
                    newton_step[r, c] = (
                        -1 * np.linalg.lstsq(DR, R_list[r].reshape(-1, 1), rcond=None)[0].ravel()
                    )
        return newton_step, R

    def _shift_reference_set(self):
        """shift the reference set when the following conditions are True:
        1. always shift the first reference set (`self.iter_count == 0`); Otherwise, weird matching can happen
        2. if at least one approximation point is close to its matched target and the Newton step is not zero.
        """
        distance = np.linalg.norm(self.state.Y - self._igd._medoids, axis=1)
        masks = (
            np.array([True] * self.N)
            if self.iter_count == 0
            else np.bitwise_and(
                np.isclose(distance, 0),
                np.isclose(np.linalg.norm(self.step[:, : self.dim_p], axis=1), 0),
            )
        )
        # import matplotlib.pyplot as plt

        # fig, ax = plt.subplots(1, 1, figsize=(25, 6.5))
        # plt.subplots_adjust(right=0.93, left=0.05)
        # medoids = self._igd._medoids
        # ax.plot(medoids[:, 0], medoids[:, 1], "k.", alpha=0.5)
        # ax.plot(self.state.Y[:, 0], self.state.Y[:, 1], "k+", alpha=0.5)
        # for i, m in enumerate(medoids):
        #     ax.plot((m[0], self.state.Y[i, 0]), (m[1], self.state.Y[i, 1]), "r--", alpha=0.5)
        # plt.savefig(f"{self.iter_count}.pdf", dpi=1000)

        indices = np.nonzero(masks)[0]
        if len(indices) == 0:
            return

        if self._eta is None:
            self._eta = dict()
            # compute the shift direction with CHIM
            for i in range(self.n_cluster):
                Y = self.state.Y[self.Y_idx[i]]
                idx = non_domin_sort(Y, only_front_indices=True)[0]
                self._eta[i] = compute_chim(Y[idx])

        # shift the medoids
        for i, k in enumerate(indices):
            n = self._eta[self.Y_label[k]]
            # NOTE: initial shift CF1: 0.6, CF2/3: 0.2
            # DTLZ4: 0.08 seems to work a bit better
            # TODO: create a configuration class to set those hyperparameter of this method, e.g., shift amount
            v = 0.05 * n if self.iter_count > 0 else 0.08 * n  # the initial shift is a bit larger
            self._igd.shift_medoids(v, k)

        if self.iter_count == 0:  # record the initial medoids
            self.history_medoids = [[m.copy()] for m in self._igd._medoids]
        else:
            # log the updated medoids
            for i, k in enumerate(indices):
                self.history_medoids[k].append(self._igd._medoids[indices][i])
        self.logger.info(f"{len(indices)} target points are shifted")

    def _backtracking_line_search(
        self, step: np.ndarray, R: np.ndarray, max_step_size: np.ndarray = None
    ) -> float:
        """backtracking line search with Armijo's condition"""
        c1 = 1e-4
        if np.all(np.isclose(step, 0)):
            return np.ones((self.N, 1))

        def phi_func(alpha, i):
            state = deepcopy(self.state)
            x = state.X[i].copy()
            x += alpha * step[i]
            state.update_one(x, i)
            self.active_indicator.re_match = False
            R_ = self._compute_R(state)[0][i]
            self.active_indicator.re_match = True
            self.state.n_jac_evals = state.n_jac_evals
            return np.linalg.norm(R_)

        if max_step_size is not None:
            step_size = max_step_size.reshape(-1, 1)
        else:
            step_size = np.ones((self.N, 1))
        for i in range(self.N):
            phi = [np.linalg.norm(R[i])]
            s = [0, step_size[i]]
            for _ in range(8):
                phi.append(phi_func(s[-1], i))
                # Armijo–Goldstein condition
                # when R norm is close to machine precision, it makes no sense to perform the line search
                success = phi[-1] <= (1 - c1 * s[-1]) * phi[0] or np.isclose(phi[0], np.finfo(float).eps)
                if success:
                    break
                else:
                    if 1 < 2:
                        # cubic interpolation to compute the next step length
                        d1 = -phi[-2] - phi[-1] - 3 * (phi[-2] - phi[-1]) / (s[-2] - s[-1])
                        d2 = np.sign(s[-1] - s[-2]) * np.sqrt(d1**2 - phi[-2] * phi[-1])
                        s_ = s[-1] - (s[-1] - s[-2]) * (-phi[-1] + d2 - d1) / (-phi[-1] + phi[-2] + 2 * d2)
                        s_ = s[-1] * 0.5 if np.isnan(s_) else np.clip(s_, 0.4 * s[-1], 0.6 * s[-1])
                        s.append(s_)
                    else:
                        s.append(s[-1] * 0.5)
            else:
                pass
                # self.logger.info("backtracking line search failed")
            step_size[i] = s[-1]
        return step_size

    def _handle_box_constraint(self, step: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The box-constraint handler projects the Newton step onto the box boundary, preventing the
        algorithm from leaving the box. It is needed when the test function is not well-defined out of the box.
        """
        if 1 < 2:
            return step, np.ones(len(step))

        primal_vars = self.state.primal
        step_primal = step[:, : self.dim_p]
        normal_vectors = np.c_[np.eye(self.dim_p), -1 * np.eye(self.dim_p)]
        # calculate the maximal step-size
        dist = np.c_[
            np.abs(primal_vars - self.xl),
            np.abs(self.xu - primal_vars),
        ]
        v = step_primal @ normal_vectors
        s = np.array([dist[i] / np.abs(np.minimum(0, vv)) for i, vv in enumerate(v)])
        max_step_size = np.array([min(1.0, np.nanmin(_)) for _ in s])
        # project Newton's direction onto the box boundary
        idx = max_step_size == 0
        if np.any(idx) > 0:
            proj_dim = [np.argmin(_) for _ in s[idx]]
            proj_axis = normal_vectors[:, proj_dim]
            step_primal[idx] -= (np.einsum("ij,ji->i", step_primal[idx], proj_axis) * proj_axis).T
            step[:, : self.dim_p] = step_primal
            # re-calculate the `max_step_size` for projected directions
            v = step[:, : self.dim_p] @ normal_vectors
            s = np.array([dist[i] / np.abs(np.minimum(0, vv)) for i, vv in enumerate(v)])
            max_step_size = np.array([min(1, np.nanmin(_)) for _ in s])
        return step, max_step_size
