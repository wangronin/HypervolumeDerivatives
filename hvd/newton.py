import logging
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.linalg import block_diag, solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist

from .base import State
from .delta_p import GenerationalDistance, InvertedGenerationalDistance
from .hypervolume_derivatives import HypervolumeDerivatives
from .reference_set import ReferenceSet
from .utils import compute_chim, get_logger, non_domin_sort, precondition_hessian, set_bounds

__authors__ = ["Hao Wang"]

np.seterr(divide="ignore", invalid="ignore")


def matrix_to_Nd_vector(X: np.ndarray, dim_primal: int, active_indices: np.ndarray) -> np.ndarray:
    return np.r_[X[:, :dim_primal].reshape(-1, 1), X[:, dim_primal:][active_indices].reshape(-1, 1)]


def Nd_vector_to_matrix(
    x: np.ndarray, N: int, dim: int, dim_primal: int, active_indices: np.ndarray
) -> np.ndarray:
    if dim == dim_primal:  # the unconstrained case
        return x.reshape(N, -1)
    active_indices = np.c_[np.array([[True] * dim_primal] * N), active_indices]
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
        metrics: Dict[str, Callable] = dict(),
        preconditioning: bool = False,
    ):
        self.dim_p: int = n_var  # the number of primal variables
        self.n_obj: int = n_obj  # the number of objectives
        self.N: int = N  # the population size
        self.xl: np.ndarray = xl
        self.xu: np.ndarray = xu
        self.ref: np.ndarray = ref
        self._check_constraints(h, g)
        self.state: State = State(
            self.dim_p, self.n_eq, self.n_ieq, func, jac, h, h_jac, h_hessian, g, g_jac, g_hessian
        )
        self.indicator = HypervolumeDerivatives(self.dim_p, self.n_obj, ref, func, jac, hessian)
        self._initialize(X0)
        self._set_logging(verbose)
        self.xtol: float = xtol
        self.max_iters: int = self.N * 10 if max_iters is None else max_iters
        self.stop_dict: Dict[str, float] = {}
        self.metrics: Dict[str, Callable] = metrics
        self.preconditioning: bool = preconditioning
        self.verbose: bool = verbose
        self.eps: float = 1e-3 * np.max(self.xu - self.xl)

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
        # TODO: maybe try initializing the multipliers to positive values for
        self.state.update(np.c_[X0, np.zeros((self.N, self.dim_d))])  # (mu, dim)
        self.iter_count: int = 0
        self._max_HV = np.product(self.ref)  # TODO: this should be moved to the `HV` class

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

    def newton_iteration(self):
        """Notes on the implementation:
        case (1): All points are infeasible, then non-dominated ones maximize HV under constraints;
            the dominated ones only optimize feasibility as proven in
            [WED+23] Wang, Hao, Michael Emmerich, André Deutz, Víctor Adrián Sosa Hernández,
            and Oliver Schütze. "The Hypervolume Newton Method for Constrained Multi-Objective
            Optimization Problems." Mathematical and Computational Applications 28, no. 1 (2023): 10.
        case (2): Some points are numerically feasible (|h(x)| < 1e-4 and g(x) <= -1e-4). Then we perform
            non-dominated sorting among the feasible ones to ensure feasible ones have non-zero Newton steps;
            and we merge the infeasible ones with the first dominance layer of the feasible, where the
            behavior of the infeasible ones falls back to case (1)
        """
        # check for anomalies in `X` and `Y`
        self._check_points_uniqueness()
        # first compute the current indicator value
        self._compute_indicator_value()
        # partition the approximation set to by feasibility
        feasible_mask = self.state.is_feasible()
        feasible_idx, infeasible_idx = np.nonzero(feasible_mask)[0], np.nonzero(~feasible_mask)[0]
        partitions = {0: np.array(range(self.N))}
        if len(feasible_idx) > 0:
            # non-dominatd sorting of the feasible points
            partitions = non_domin_sort(self.state.Y[feasible_idx], only_front_indices=True)
            partitions = {k: feasible_idx[v] for k, v in partitions.items()}
            # merge infeasible points into the first dominance layer
            partitions.update({0: np.sort(np.r_[partitions[0], infeasible_idx])})
        # compute the Newton direction for each partition
        self.step = np.zeros((self.N, self.dim))
        self.step_size = np.ones(self.N)
        self.R = np.zeros((self.N, self.dim))
        for idx in partitions.values():
            # compute Newton step
            newton_step, R = self._compute_netwon_step(self.state[idx])
            # constrain the search steps within the search box
            newton_step, max_step_size = self._handle_box_constraint(newton_step, self.state[idx])
            self.step[idx, :] = newton_step
            self.R[idx, :] = R
            # backtracking line search with Armijo's condition for each layer
            self.step_size[idx] = self._backtracking_line_search(
                self.state[idx], newton_step, R, max_step_size
            )
        # Newton iteration and evaluation
        self.state.update(self.state.X + self.step * self.step_size.reshape(-1, 1))

    def log(self):
        # TODO: maybe we should log the initial population
        self.iter_count += 1
        self.history_X += [self.state.primal.copy()]
        self.history_Y += [self.state.Y.copy()]
        self.history_indicator_value += [self.curr_indicator_value]
        self.history_R_norm += [np.median(np.linalg.norm(self.R, axis=1))]
        # compute the performance metrics
        for name, func in self.metrics.items():
            self.history_metrics[name].append(func.compute(Y=self.state.Y))
        if self.verbose:
            self.logger.info(f"iteration {self.iter_count} ---")
            self.logger.info(f"HV: {self.curr_indicator_value}")
            self.logger.info(f"step size: {self.step_size.ravel()}")
            self.logger.info(f"R norm: {self.history_R_norm[-1]}")

    def terminate(self) -> bool:
        if self.iter_count >= self.max_iters:
            self.stop_dict["iter_count"] = self.iter_count

        return bool(self.stop_dict)

    def _compute_R(
        self, state: State, grad: np.ndarray = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """compute the root-finding problem R
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
                (R, H, active_indices) -> the rooting-finding problem,
                the Jacobian of the equality constraints, and
                the indices of the active dual variables
        """
        if grad is None:
            grad = self.indicator.compute_derivatives(state.primal, state.Y, False, state.J)
        # the unconstrained case
        R, dH, active_indices = grad, None, None
        if self._constrained:
            R = np.zeros((state.N, self.dim))  # the root-finding problem
            func = lambda g, dual, h: g + np.einsum("j,jk->k", dual, h)
            cstr_value, active_indices, dH = state.cstr_value, state.active_indices, state.cstr_grad
            # only take Jacobian of the active constraints
            dH = [dH[i, k, :] for i, k in enumerate(active_indices)]
            for i, k in enumerate(active_indices):
                R[i, : self.dim_p] = func(grad[i], state.dual[i, k], dH[i])
                R[i, self.dim_p :][k] = cstr_value[i, k]
        return R, dH, active_indices

    def _compute_netwon_step(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        grad, DR = self.indicator.compute_derivatives(X=state.primal, Y=state.Y, YdX=state.J)
        R, H, idx = self._compute_R(state, grad=grad)
        # in case the Hessian is not NSD
        if self.preconditioning:
            DR = -1.0 * precondition_hessian(-1.0 * DR)
        if self._constrained:
            H = block_diag(*H)  # (N * p, N * dim), `p` is the number of active constraints
            B = state.cstr_hess
            M = block_diag(*[np.einsum("i...,i", B[i, k], state.dual[i, k]) for i, k in enumerate(idx)])
            Z = np.zeros((len(H), len(H)))
            DR = np.r_[np.c_[DR + M, H.T], np.c_[H, Z]]
        # the vector-format of R
        R_vec = matrix_to_Nd_vector(R, self.dim_p, idx)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                newton_step_ = -1 * spsolve(csc_matrix(DR), csc_matrix(R_vec))
            except:  # if DR is singular, then use the pseudoinverse
                newton_step_ = -1 * np.linalg.lstsq(DR, R_vec, rcond=None)[0].ravel()
        # convert the vector-format of the newton step to matrix format
        newton_step = Nd_vector_to_matrix(newton_step_.ravel(), state.N, self.dim, self.dim_p, idx)
        return newton_step, R

    def _compute_indicator_value(self):
        self.curr_indicator_value = self.indicator.compute(Y=self.state.Y)

    def _backtracking_line_search(
        self, state: State, step: np.ndarray, R: np.ndarray, max_step_size: float = 1
    ) -> float:
        """backtracking line search with Armijo's condition"""
        c1 = 1e-5
        if np.any(np.isclose(np.median(step[:, : self.dim_p]), np.finfo(np.double).resolution)):
            return max_step_size

        def phi_func(alpha: float) -> float:
            state_ = deepcopy(state)
            state_.update(state.X + alpha * step)
            R = self._compute_R(state_)[0]
            return np.linalg.norm(R)

        step_size = max_step_size
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

    def _handle_box_constraint(self, step: np.ndarray, state: State) -> Tuple[np.ndarray, np.ndarray]:
        primal_vars, step_primal = state.primal, step[:, : self.dim_p]
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
        step[:, : self.dim_p] = step_primal
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
        ref: ReferenceSet,
        xl: Union[List[float], np.ndarray],
        xu: Union[List[float], np.ndarray],
        type: str = "deltap",
        N: int = 5,
        h: Callable = None,
        h_jac: callable = None,
        h_hessian: callable = None,
        g: Callable = None,
        g_jac: Callable = None,
        g_hessian: callable = None,
        x0: np.ndarray = None,
        max_iters: Union[int, str] = np.inf,
        xtol: float = 0,
        verbose: bool = True,
        metrics: Dict[str, Callable] = dict(),
        preconditioning: bool = False,
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
        self.preconditioning: bool = preconditioning
        self._check_constraints(h, g)
        self.ref: ReferenceSet = ref
        self.state = State(
            self.dim_p, self.n_eq, self.n_ieq, func, jac, h, h_jac, h_hessian, g, g_jac, g_hessian
        )
        self._initialize(x0)
        self._set_indicator(self.ref, func, jac, hessian)
        self._set_logging(verbose)
        # parameters of the stop criteria
        self.xtol: float = xtol
        self.max_iters: int = self.N * 10 if max_iters is None else max_iters
        self.stop_dict: Dict[str, float] = {}
        self.metrics = metrics
        self.preconditioning: bool = preconditioning

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
            if 11 < 2:
                # NOTE: ad-hoc solution for CF2 and IDTLZ1 since the Jacobian on the box boundary is
                # not defined on the decision boundary or the local Hessian is ill-conditioned.
                X0 = np.clip(X0 - self.xl, 1e-2, 1) + self.xl
                X0 = np.clip(X0 - self.xu, -1, -1e-2) + self.xu
            self.N = len(X0)
        else:
            # sample `x` u.a.r. in `[lb, ub]`
            assert self.N is not None
            assert all(~np.isinf(self.xl)) & all(~np.isinf(self.xu))
            X0 = np.random.rand(self.N, self.dim_p) * (self.xu - self.xl) + self.xl  # (mu, dim_primal)
        # initialize the state variables
        self.state.update(np.c_[X0, np.zeros((self.N, self.dim_d))])  # (mu, dim)
        self.iter_count: int = 0

    def _set_indicator(self, ref: ReferenceSet, func: Callable, jac: Callable, hessian: Callable):
        self._gd = GenerationalDistance(ref, func, jac, hessian)
        self._igd = InvertedGenerationalDistance(ref, func, jac, hessian, matching=True)

    def _set_logging(self, verbose):
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
        self.step, self.R = self._compute_netwon_step(self.state)
        # prevent the decision points from moving out of the decision space. Needed for CF7 with NSGA-III
        self.step, max_step_size = self._handle_box_constraint(self.step)
        # backtracking line search for the step size
        self.step_size = self._backtracking_line_search(self.step, self.R, max_step_size)
        # Newton iteration and evaluation
        self.state.update(self.state.X + self.step_size * self.step)

    def log(self):
        self.iter_count += 1
        self.history_Y += [self.state.Y.copy()]
        self.history_X += [self.state.primal.copy()]
        self.history_indicator_value += [self.curr_indicator_value]
        self.history_R_norm += [np.median(np.linalg.norm(self.R, axis=1))]
        if self.verbose:
            self.logger.info(f"iteration {self.iter_count} ---")
            self.logger.info(f"{self.active_indicator.__class__.__name__}: {self.curr_indicator_value}")
            self.logger.info(f"step size: {self.step_size.ravel()}")
            self.logger.info(f"R norm: {self.history_R_norm[-1]}")
        # compute the performance metrics
        for name, func in self.metrics.items():
            value = func.compute(Y=self.state.Y)
            self.history_metrics[name].append(value)
            self.logger.info(f"{name}: {value}")

    def terminate(self) -> bool:
        if self.iter_count >= self.max_iters:
            self.stop_dict["iter_count"] = self.iter_count
        return bool(self.stop_dict)

    def _compute_indicator_value(self, Y: np.ndarray):
        self.GD_value = self._gd.compute(Y=Y)
        self.IGD_value = self._igd.compute(Y=Y)

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
                X=primal_vars, Y=state.Y, compute_hessian=False, Jacobian=state.J
            )
        R = grad  # the unconstrained case
        dH, active_indices = None, np.array([[True] * state.n_var] * self.N)
        if self._constrained:
            func = lambda g, dual, h: g + np.einsum("j,jk->k", dual, h)
            v, idx, dH = state.cstr_value, state.active_indices, state.cstr_grad
            dH = [dH[i][idx, :] for i, idx in enumerate(idx)]
            R = [np.r_[func(grad[i], dual_vars[i, k], dH[i]), v[i, k]] for i, k in enumerate(idx)]
            active_indices = np.c_[active_indices, idx]
        return R, dH, active_indices

    def _compute_netwon_step(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        newton_step = np.zeros((self.N, self.dim))  # Netwon steps
        R = np.zeros((self.N, self.dim))  # the root-finding problem
        # gradient and Hessian of the incumbent indicator
        grad, Hessian = self.active_indicator.compute_derivatives(X=state.primal, Y=state.Y, Jacobian=state.J)
        # the root-finding problem and the gradient of the active constraints
        R_list, dH, active_indices = self._compute_R(state, grad=grad)
        idx = active_indices[:, self.dim_p :]
        if state.cstr_hess.size != 0:
            S = [np.einsum("i...,i", state.cstr_hess[i, k], state.dual[i, k]) for i, k in enumerate(idx)]
        else:
            S = [np.zeros((self.dim_p, self.dim_p))] * self.N
        # compute the Newton step for each approximation point - lower computation costs
        for r in range(self.N):
            c = active_indices[r]
            dh = np.array([]) if dH is None else dH[r]
            Z = np.zeros((len(dh), len(dh)))
            # pre-condition indicator's Hessian if needed, e.g., on ZDT6, CF1, CF7
            if self.preconditioning:
                Hessian[r] = precondition_hessian(Hessian[r])
            # derivative of the root-finding problem
            DR = np.r_[np.c_[Hessian[r] + S[r], dh.T], np.c_[dh, Z]] if self._constrained else Hessian[r]
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
        if self.iter_count == 0:  # TODO: maybe do not perform the initial shift here..
            masks = np.array([True] * self.N)
        else:
            distance = np.linalg.norm(self.state.Y - self.ref.reference_set, axis=1)
            step_len = np.linalg.norm(self.step[:, : self.dim_p], axis=1)
            masks = np.bitwise_and(np.isclose(distance, 0), np.isclose(step_len, 0))
        indices = np.nonzero(masks)[0]
        # the initial shift is a bit larger
        self.ref.shift(0.08 if self.iter_count == 0 else 0.05, indices)
        for k in indices:  # log the updated medoids
            self.history_medoids[k].append(self.ref.reference_set[k].copy())
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
        # NOTE: for ZDT6, we have to project the gradient
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
