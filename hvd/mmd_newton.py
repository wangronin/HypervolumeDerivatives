import logging
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.linalg import solve

from .base import State
from .mmd import MMDMatching
from .reference_set import ClusteredReferenceSet
from .utils import get_logger, precondition_hessian, set_bounds

np.seterr(divide="ignore", invalid="ignore")

__authors__ = ["Hao Wang"]


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
        ref: ClusteredReferenceSet,
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
        self.dim_p = n_var
        self.n_obj = n_obj
        self.N = N
        self.xl = xl
        self.xu = xu
        self.ref = ref
        self._check_constraints(h, g)
        self.state = State(self.dim_p, self.n_eq, self.n_ieq, func, jac, h, h_jac, g, g_jac)
        self.indicator = MMDMatching(
            self.dim_p, self.n_obj, ref=self.ref, func=func, jac=jac, hessian=hessian, theta=1.0, beta=0.5
        )
        self._initialize(X0)
        self._set_logging(verbose)
        # parameters controlling stop criteria
        self.xtol = xtol
        self.max_iters: int = self.N * 10 if max_iters is None else max_iters
        self.stop_dict: Dict = {}
        self.metrics = metrics

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
            # X0 = np.clip(X0 - self.xl, 1e-5, 1) + self.xl
            # X0 = np.clip(X0 - self.xu, -1, -1e-5) + self.xu
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
        self.step_size = self._backtracking_line_search(self.step, self.R, max_step_size)
        # Newton iteration and evaluation
        self.state.update(self.state.X + self.step_size * self.step)

    def log(self):
        self.iter_count += 1
        self.history_Y += [self.state.Y.copy()]
        self.history_X += [self.state.primal.copy()]
        self.history_indicator_value += [self.curr_indicator_value]
        self.history_R_norm += [np.median(np.linalg.norm(self.R, axis=1))]
        for name, func in self.metrics.items():  # compute the performance metrics
            self.history_metrics[name].append(func.compute(Y=self.state.Y))
        if self.verbose:
            self.logger.info(f"iteration {self.iter_count} ---")
            self.logger.info(f"{self.indicator.__class__.__name__}: {self.curr_indicator_value}")
            self.logger.info(f"step size: {self.step_size.ravel()}")
            self.logger.info(f"R norm: {self.history_R_norm[-1]}")

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
                (R, H, idx) -> the rooting-finding problem,
                the Jacobian of the equality constraints, and
                the indices that are active among primal-dual variables
        """
        primal_vars, dual_vars = state.primal, state.dual
        if grad is None:
            grad = self.indicator.compute_derivatives(
                X=primal_vars, Y=state.Y, compute_hessian=False, jacobian=state.J
            )
        R = grad  # the unconstrained case
        dH, idx = None, None
        if self._constrained:
            func = lambda g, dual, h: g + np.einsum("j,jk->k", dual, h)
            v, idx, dH = state.cstr_value, state.active_indices, state.dH
            dH = [dH[i][idx, :] for i, idx in enumerate(idx)]
            R = [np.r_[func(grad[i], dual_vars[i, k], dH[i]), v[i, k]] for i, k in enumerate(idx)]
        return R, dH, idx

    def _compute_netwon_step(self) -> Tuple[np.ndarray, np.ndarray]:
        primal_vars = self.state.primal
        newton_step = np.zeros((self.N, self.dim))  # Netwon steps
        R = np.zeros((self.N, self.dim))  # the root-finding problem
        # gradient and Hessian of the incumbent indicator
        grad, Hessian = self.indicator.compute_derivatives(
            X=primal_vars,
            Y=self.state.Y,
            jacobian=self.state.J,
        )
        # the root-finding problem and the gradient of the active constraints
        R_list, dH, active_indices = self._compute_R(self.state, grad=grad)
        Hessian = precondition_hessian(Hessian)
        DR, R = Hessian, grad
        # TODO: implement the constrained case
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                # TODO: use the sparse matrix representation to save some time here
                newton_step = -1 * solve(DR, R.reshape(-1, 1)).reshape(self.N, -1)
            except:
                # if DR is singular, then use the pseudoinverse.
                newton_step = -1 * np.linalg.lstsq(DR, R.reshape(-1, 1), rcond=None)[0].reshape(self.N, -1)
        return newton_step, R

    def _shift_reference_set(self):
        """shift the reference set when the following conditions are True:
        1. always shift the first reference set (`self.iter_count == 0`); Otherwise, weird matching can happen
        2. if at least one approximation point is close to its matched target and the Newton step is not zero.
        """
        distance = np.linalg.norm(self.state.Y - self.ref.medoids, axis=1)
        if self.iter_count == 0:
            masks = np.array([True] * self.N)
        else:
            masks = np.bitwise_and(
                np.isclose(distance, 0),
                np.isclose(np.linalg.norm(self.step[:, : self.dim_p], axis=1), 0),
            )
        indices = np.nonzero(masks)[0]
        self.ref.shift(0.05, indices)
        self.indicator.ref = self.ref  # TODO: check if this is needed
        # log the updated medoids
        for k in indices:
            self.history_medoids[k].append(self.ref.medoids[k].copy())
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
            x = state.X[i]
            x += alpha * step[i]
            state.update_one(x, i)
            self.indicator.re_match = False
            R_ = self._compute_R(state)[0][i]
            self.indicator.re_match = True
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
        NOTE: this function is experimental
        """
        if 11 < 2:
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
