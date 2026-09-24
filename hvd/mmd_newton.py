import logging
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.linalg import block_diag, solve
from scipy.spatial.distance import cdist

from .base import State
from .mmd import MMD, MMDMatching
from .utils import (
    Nd_vector_to_matrix,
    get_logger,
    matrix_to_Nd_vector,
    project_box_step,
    regularize_hessian_block,
    set_bounds,
)


class MMDN:
    """MMD Newton method

    Newton-Raphson method driven by an MMD indicator and its derivatives.
    """

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        func: callable,
        jac: callable,
        xl: Union[List[float], np.ndarray],
        xu: Union[List[float], np.ndarray],
        indicator: MMD | MMDMatching,
        N: int = 5,
        h: Callable = None,
        h_jac: callable = None,
        h_hessian: callable = None,
        g: Callable = None,
        g_jac: Callable = None,
        g_hessian: callable = None,
        X0: np.ndarray = None,
        max_iters: Union[int, str] = np.inf,
        xtol: float = 0,
        verbose: bool = True,
        metrics: Dict[str, Callable] = dict(),
        regularization: bool = False,
        project_box_constraints: bool = False,
    ) -> None:
        """
        Args:
            n_var (int): dimensionality of the search space.
            n_obj (int): number of objectives.
            func (callable): the objective function to be minimized.
            jac (callable): the Jacobian of objectives, should return a matrix of size (n_objective, dim)
            xl (Union[List[float], np.ndarray]): the lower bound of search variables.
                When it is not a `float`, it must have shape (dim, ).
            xu (Union[List[float], np.ndarray]): the upper bound of search variables.
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
            project_box_constraints (bool, optional): project outward search
                directions and limit steps to remain inside the decision box.
                Defaults to False.
            indicator (MMD or MMDMatching): an indicator configured for this problem,
                including its objective callbacks, reference set, and kernel.
        """
        self.dim_p: int = n_var
        self.n_obj: int = n_obj
        self.N: int = N
        self.xl: np.ndarray = xl
        self.xu: np.ndarray = xu
        self.indicator = indicator
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
        self._initialize(X0)
        self._set_logging(verbose)
        # parameters of the stop criteria
        self.xtol: float = xtol
        self.max_iters: int = self.N * 10 if max_iters is None else max_iters
        self.stop_dict: Dict[str, float] = {}
        self.metrics: Dict[str, Callable] = metrics
        self.regularization: bool = regularization
        self.project_box_constraints: bool = project_box_constraints

    def _check_constraints(self, h: Callable | None, g: Callable | None) -> None:
        # An absent family is either an omitted callback or a None result.
        x = np.random.rand(self.dim_p) * (self.xu - self.xl) + self.xl
        H = None if h is None else h(x)
        G = None if g is None else g(x)
        self.n_eq = 0 if H is None else np.size(H)
        self.n_ieq = 0 if G is None else np.size(G)
        self.dim_d = self.n_eq + self.n_ieq
        self.dim = self.dim_p + self.dim_d
        self._constrained = self.dim_d > 0

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

    def _set_logging(self, verbose: bool):
        """parameters for logging the history"""
        self.verbose: bool = verbose
        self.curr_indicator_value: float = None
        self.history_Y: List[np.ndarray] = []
        self.history_X: List[np.ndarray] = []
        self.history_indicator_value: List[float] = []
        self.history_R_norm: List[float] = []
        self.history_metrics: Dict[str, List] = defaultdict(list)
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

    def run(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        while not self.terminate():
            self.newton_iteration()
            self.log()
        return self.state.primal, self.state.Y, self.stop_dict

    def newton_iteration(self):
        # evaluate first so matching indicators establish their current targets
        self.curr_indicator_value = self.indicator.compute(Y=self.state.Y)
        # initial shifting of reference point or if the reference is very close to the approximation point
        self._shift_reference_set()
        # compute the Newton step
        self.step, self.R = self._compute_netwon_step()
        max_step_size = None
        if self.project_box_constraints:
            self.step, max_step_size = project_box_step(self.step, self.state.primal, self.xl, self.xu)
        # backtracking line search for the step size
        self.step_size = self._backtracking_line_search_individual(self.step, self.R, max_step_size)
        # Newton iteration and evaluation
        self.state.update(self.state.X + self.step_size.reshape(-1, 1) * self.step)
        self.iter_count += 1

    def log(self):
        self.history_Y += [self.state.Y.copy()]
        self.history_X += [self.state.primal.copy()]
        self.history_indicator_value += [self.curr_indicator_value]
        self.history_R_norm += [np.median(np.linalg.norm(self.R, axis=1))]
        if self.verbose:
            self.logger.info(f"iteration {self.iter_count} ---")
            self.logger.info(f"{self.indicator.__class__.__name__}: {self.curr_indicator_value}")
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

    def _shift_reference_set(self) -> None:
        """Request a shift initially, then for points that reach a target and stop moving.

        Newton controls the trigger using its previous primal step and the
        indicator's current targets. The indicator performs and records the shift.
        """
        if self.iter_count == 0:
            indices = np.arange(self.N)
        else:
            distance = np.min(cdist(self.state.Y, self.indicator.ref.reference_set), axis=1)
            step_norm = np.linalg.norm(self.step[:, : self.dim_p], axis=1)
            indices = np.flatnonzero(np.isclose(distance, 0) & np.isclose(step_norm, 0))
        if len(indices):
            self.indicator.shift_reference_set(indices=indices)

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
        primal_vars, dual_vars = state.primal, state.dual
        if grad is None:
            grad = self.indicator.compute_derivatives(
                X=primal_vars, Y=state.Y, compute_hessian=False, jacobian=state.J
            )
        R, dH, active_indices = grad, None, None
        if self._constrained:
            R = np.zeros((state.N, self.dim))  # the root-finding problem
            func = lambda g, dual, h: g + np.einsum("j,jk->k", dual, h)
            cstr_value, active_indices, dH = state.cstr_value, state.active_indices, state.cstr_grad
            dH = [dH[i, k, :] for i, k in enumerate(active_indices)]
            for i, k in enumerate(active_indices):
                R[i, : self.dim_p] = func(grad[i], dual_vars[i, k], dH[i])
                R[i, self.dim_p :][k] = cstr_value[i, k]
        return R, dH, active_indices

    def _compute_netwon_step(self) -> Tuple[np.ndarray, np.ndarray]:
        grad, Hessian = self.indicator.compute_derivatives(
            X=self.state.primal, Y=self.state.Y, jacobian=self.state.J
        )
        if self.regularization:  # sometimes the Hessian is not PSD
            Hessian = regularize_hessian_block(Hessian, block_size=self.dim_p)
        R, dH, idx = self._compute_R(self.state, grad=grad)
        DR = Hessian
        if self._constrained:
            dH = block_diag(*dH)  # (N * p, N * dim), `p` is the number of active constraints
            Z = np.zeros((len(dH), len(dH)))
            B = self.state.cstr_hess
            S = block_diag(*[np.einsum("i...,i", B[i, k], self.state.dual[i, k]) for i, k in enumerate(idx)])
            DR = np.r_[np.c_[DR + S, dH.T], np.c_[dH, Z]]
        # vectorize R
        R_ = matrix_to_Nd_vector(R, self.dim_p, idx)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                # NOTE: use sparse matrix operations here does not save much time
                newton_step_ = -1 * solve(DR, R_)
            except:  # if DR is singular, then use the pseudoinverse
                newton_step_ = -1 * np.linalg.lstsq(DR, R_, rcond=None)[0]
        # convert the vector-format of the newton step to matrix format
        newton_step = Nd_vector_to_matrix(newton_step_.ravel(), self.N, self.dim, self.dim_p, idx)
        return newton_step, R

    def _backtracking_line_search_global(
        self, step: np.ndarray, R: np.ndarray, max_step_size: np.ndarray = None
    ) -> float:
        """backtracking line search with Armijo's condition. Global step-size control"""
        c1 = 1e-6
        if 1 < 2 and np.any(np.isclose(np.median(step[:, : self.dim_p]), np.finfo(np.double).resolution)):
            return np.array([1])

        def phi_func(alpha):
            state_ = deepcopy(self.state)
            state_.update(state_.X + alpha * step)
            with self.indicator.trial_evaluation():
                R_ = self._compute_R(state_)[0]
            return np.linalg.norm(R_)

        step_size = min(max_step_size) if max_step_size is not None else 1
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
                s.append(s[-1] * 0.5)
        else:
            self.logger.warn("backtracking line search failed")
        step_size = s[-1]
        return step_size

    def _backtracking_line_search_individual(
        self, step: np.ndarray, R: np.ndarray, max_step_size: np.ndarray = None
    ) -> np.ndarray:
        # TODO: use the backtracking line search in scipy
        """backtracking line search with Armijo's condition"""
        c1 = 1e-4
        if 1 < 2 and np.all(np.isclose(step, 0)):
            return np.ones((self.N, 1))

        def phi_func(alpha, i):
            state = deepcopy(self.state)
            x = state.X[i].copy()
            x += alpha * step[i]
            state.update_one(x, i)
            # this step takes too long since we compute the gradient at all points while only one changes
            with self.indicator.trial_evaluation():
                R_ = self._compute_R(state)[0][i]
            self.state.n_jac_evals = state.n_jac_evals
            return np.linalg.norm(R_)

        step_size = max_step_size if max_step_size is not None else np.ones(self.N)
        for i in range(self.N):
            phi = [np.linalg.norm(R[i])]
            s = [0, step_size[i]]
            for _ in range(6):
                phi.append(phi_func(s[-1], i))
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
                        s.append(s[-1] / 2)
            else:
                self.logger.warning("backtracking line search failed")
                step_size[i] = 0.0
                continue
            step_size[i] = s[-1]
        return step_size
