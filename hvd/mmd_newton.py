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
from .reference_set import ReferenceSet
from .utils import get_logger, precondition_hessian, set_bounds


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


class MMDNewton:
    """MMD Newton method

    Newton-Raphson method to minimize the modified MMD indicator
    """

    def __init__(
        self,
        n_var: int,
        n_obj: int,
        func: callable,
        jac: callable,
        hessian: callable,
        ref: ReferenceSet,
        xl: Union[List[float], np.ndarray],
        xu: Union[List[float], np.ndarray],
        N: int = 5,
        h: Callable = None,
        h_jac: callable = None,
        g: Callable = None,
        g_jac: Callable = None,
        X0: np.ndarray = None,
        max_iters: Union[int, str] = np.inf,
        xtol: float = 0,
        verbose: bool = True,
        metrics: Dict[str, Callable] = dict(),
        preconditioning: bool = False,
        matching: bool = True,
        **kwargs,
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
        self.dim_p: int = n_var
        self.n_obj: int = n_obj
        self.N: int = N
        self.xl: np.ndarray = xl
        self.xu: np.ndarray = xu
        self.ref: ReferenceSet = ref  # TODO: we should pass ref to the indicator directly
        self.matching: bool = matching
        self._check_constraints(h, g)
        self.state = State(self.dim_p, self.n_eq, self.n_ieq, func, jac, h=h, h_jac=h_jac, g=g, g_jac=g_jac)
        if self.matching:
            self.indicator = MMDMatching(
                self.dim_p, self.n_obj, self.ref, func, jac, hessian, beta=0.25, **kwargs
            )
        else:
            self.indicator = MMD(self.dim_p, self.n_obj, self.ref, func, jac, hessian, **kwargs)
        self._initialize(X0)
        self._set_logging(verbose)
        # parameters of the stop criteria
        self.xtol: float = xtol
        self.max_iters: int = self.N * 10 if max_iters is None else max_iters
        self.stop_dict: Dict[str, float] = {}
        self.metrics: Dict[str, Callable] = metrics
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
        # prevent the decision points from moving out of the decision space.
        self.step, max_step_size = self._handle_box_constraint(self.step)
        # backtracking line search for the step size
        self.step_size = self._backtracking_line_search_global(self.step, self.R, max_step_size)
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

    def _compute_indicator_value(self, Y: np.ndarray):
        self.curr_indicator_value = self.indicator.compute(Y=Y)

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
        primal_vars, dual_vars = state.primal, state.dual
        if grad is None:
            grad = self.indicator.compute_derivatives(
                X=primal_vars, Y=state.Y, compute_hessian=False, jacobian=state.J
            )
        R = grad  # the unconstrained case
        dH, active_indices = None, np.array([[True] * state.n_var] * state.N)
        if self._constrained:
            R = np.zeros((state.N, self.dim))  # the root-finding problem
            func = lambda g, dual, h: g + np.einsum("j,jk->k", dual, h)
            cstr_value, idx, dH = state.cstr_value, state.active_indices, state.cstr_grad
            dH = [dH[i, idx, :] for i, idx in enumerate(idx)]  # only take Jacobian of the active constraints
            active_indices = np.c_[active_indices, idx]
            for i, k in enumerate(active_indices):
                R[i, k] = np.r_[func(grad[i], dual_vars[i, idx[i]], dH[i]), cstr_value[i, idx[i]]]
        return R, dH, active_indices

    def _compute_netwon_step(self) -> Tuple[np.ndarray, np.ndarray]:
        grad, Hessian = self.indicator.compute_derivatives(
            X=self.state.primal, Y=self.state.Y, jacobian=self.state.J
        )
        if self.preconditioning:  # sometimes the Hessian is not PSD
            Hessian = precondition_hessian(Hessian)
        R, dH, active_indices = self._compute_R(self.state, grad=grad)
        DR = Hessian
        if self._constrained:
            dH = block_diag(*dH)  # (N * p, N * dim), `p` is the number of active constraints
            Z = np.zeros((len(dH), len(dH)))
            DR = np.r_[np.c_[DR, dH.T], np.c_[dH, Z]]
        # vectorize R
        idx = active_indices[:, self.dim_p :]
        R_ = np.r_[R[:, : self.dim_p].reshape(-1, 1), R[:, self.dim_p :][idx].reshape(-1, 1)]
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                # NOTE: use sparse matrix operations here does not save much time
                newton_step_ = -1 * solve(DR, R_)
            except:  # if DR is singular, then use the pseudoinverse
                newton_step_ = -1 * np.linalg.lstsq(DR, R_, rcond=None)[0]
        # convert the vector-format of the newton step to matrix format
        newton_step = Nd_vector_to_matrix(newton_step_.ravel(), self.N, self.dim, self.dim_p, active_indices)
        return newton_step, R

    def _shift_reference_set(self):
        """shift the reference set when the following conditions are True:
        1. always shift the first reference set (`self.iter_count == 0`); Otherwise, weird matching can happen
        2. if at least one approximation point is close to its matched target and the Newton step is not zero.
        """
        if self.iter_count == 0:  # TODO: maybe do not perform the initial shift here..
            masks = np.array([True] * self.N)
        else:
            distance = np.min(cdist(self.state.Y, self.ref.reference_set), axis=1)
            step_norm = np.linalg.norm(self.step[:, : self.dim_p], axis=1)
            masks = np.bitwise_and(np.isclose(distance, 0), np.isclose(step_norm, 0))

        indices = np.nonzero(masks)[0]
        self.ref.shift(0.08, indices)
        for k in indices:  # log the updated medoids
            self.history_medoids[k].append(self.ref.reference_set[k].copy())
        self.logger.info(f"{len(indices)} target points are shifted")

    def _backtracking_line_search_global(
        self, step: np.ndarray, R: np.ndarray, max_step_size: np.ndarray = None
    ) -> float:
        """backtracking line search with Armijo's condition. Global step-size control"""
        c1 = 1e-6
        if 11 < 2 or np.any(np.isclose(np.median(step[:, : self.dim_p]), np.finfo(np.double).resolution)):
            return np.array([1])

        def phi_func(alpha):
            state_ = deepcopy(self.state)
            state_.update(state_.X + alpha * step)
            self.indicator.re_match = False
            R_ = self._compute_R(state_)[0]
            self.indicator.re_match = True
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
        if 11 < 2 or np.all(np.isclose(step, 0)):
            return np.ones((self.N, 1))

        def phi_func(alpha, i):
            state = deepcopy(self.state)
            x = state.X[i].copy()
            x += alpha * step[i]
            state.update_one(x, i)
            self.indicator.re_match = False
            # this step takes too long since we compute the gradient at all points while only one changes
            R_ = self._compute_R(state)[0][i]
            self.indicator.re_match = True
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
                    if 1 < 2:
                        # cubic interpolation to compute the next step length
                        d1 = -phi[-2] - phi[-1] - 3 * (phi[-2] - phi[-1]) / (s[-2] - s[-1])
                        d2 = np.sign(s[-1] - s[-2]) * np.sqrt(d1**2 - phi[-2] * phi[-1])
                        s_ = s[-1] - (s[-1] - s[-2]) * (-phi[-1] + d2 - d1) / (-phi[-1] + phi[-2] + 2 * d2)
                        s_ = s[-1] * 0.5 if np.isnan(s_) else np.clip(s_, 0.4 * s[-1], 0.6 * s[-1])
                        s.append(s_)
                    else:
                        s.append(s[-1] / 2)
            else:
                self.logger.warn("backtracking line search failed")
            step_size[i] = s[-1]
        return step_size

    def _handle_box_constraint(self, step: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The box-constraint handler projects the Newton step onto the box boundary, preventing the
        algorithm from leaving the box. It is needed when the test function is not well-defined out of the box.
        NOTE: this function is experimental
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
