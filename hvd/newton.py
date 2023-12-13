import logging
import warnings
from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.linalg import block_diag, cholesky, pinvh, solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist

from .delta_p import GenerationalDistance, InvertedGenerationalDistance
from .hypervolume import hypervolume
from .hypervolume_derivatives import HypervolumeDerivatives
from .utils import compute_chim, get_logger, merge_lists, non_domin_sort, precondition_hessian, set_bounds

np.seterr(divide="ignore", invalid="ignore")

__authors__ = ["Hao Wang"]


class HVN:
    """Hypervolume Newton method

    Newton-Raphson method to maximize the hypervolume indicator, subject to equality constraints
    """

    def __init__(
        self,
        dim: int,
        n_objective: int,
        func: callable,
        jac: callable,
        hessian: callable,
        ref: Union[List[float], np.ndarray],
        lower_bounds: Union[List[float], np.ndarray],
        upper_bounds: Union[List[float], np.ndarray],
        mu: int = 5,
        h: Callable = None,
        h_jac: callable = None,
        h_hessian: callable = None,
        x0: np.ndarray = None,
        max_iters: Union[int, str] = np.inf,
        minimization: bool = True,
        xtol: float = 1e-3,
        HVtol: float = -np.inf,
        verbose: bool = True,
        problem_name: str = None,
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
            ref (Union[List[float], np.ndarray]): the reference point, of shape (n_objective, )
            lower_bounds (Union[List[float], np.ndarray], optional): the lower bound of search variables.
                When it is not a `float`, it must have shape (dim, ).
            upper_bounds (Union[List[float], np.ndarray], optional): The upper bound of search variables.
                When it is not a `float`, it must have must have shape (dim, ).
            mu (int, optional): the approximation set size. Defaults to 5.
            h (Callable, optional): the equality constraint function, should return a vector of shape
                (n_constraint, ). Defaults to None.
            h_jac (callable, optional): the Jacobian of constraint function,
                should return a matrix of (n_constraint, dim). Defaults to None.
            h_hessian (callable, optional): the Jacobian of constraint function.
                should return a matrix of (n_constraint, dim, dim) Defaults to None.
            x0 (np.ndarray, optional): the initial approximation set, of shape (mu, dim). Defaults to None.
            max_iters (Union[int, str], optional): maximal iterations of the algorithm. Defaults to np.inf.
            minimization (bool, optional): to minimize or maximize. Defaults to True.
            xtol (float, optional): absolute distance in the approximation set between consecutive iterations
                that is used to determine convergence. Defaults to 1e-3.
            HVtol (float, optional): absolute change in hypervolume indicator value between consecutive
                iterations that is used to determine convergence. Defaults to -np.inf.
            verbose (bool, optional): verbosity of the output. Defaults to True.
        """
        self.minimization = minimization
        self.dim_primal = dim
        self.n_objective = n_objective
        self.mu = mu
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.ref = ref
        self.problem_name = problem_name
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
            self.dim_primal,
            self.n_objective,
            ref,
            func,
            jac,
            hessian,
            minimization=minimization,
        )
        self.iter_count: int = 0
        self.max_iters = max_iters
        self.verbose: bool = verbose
        self.eps = 1e-3 * np.max(self.upper_bounds - self.lower_bounds)
        self._init_logging_var()
        self._initialize(x0)

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
            self.n_eq_cstr = 1 if isinstance(v, (int, float)) else len(v)
            # to make the Hessian of Eq. constraints always a 3D tensor
            self._h_hessian = lambda x: self.h_hessian(x).reshape(self.n_eq_cstr, self.dim_primal, -1)
            X0 = np.c_[X0, np.ones((self.mu, self.n_eq_cstr)) / self.mu]
        else:
            self.n_eq_cstr = 0

        self._get_primal_dual = lambda X: (
            X[:, : self.dim_primal],
            X[:, self.dim_primal :],
        )
        self.dim = self.dim_primal + self.n_eq_cstr
        self.X = X0
        self.Y = np.array([self.func(x) for x in self._get_primal_dual(self.X)[0]])  # (mu, n_objective)

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
        return self._get_primal_dual(self.X)[0], self.Y, self.stop_dict

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
                cholesky(-H + tau * I, lower=True)
                break
            except:
                # NOTE: the multiplier is not working for Eq1IDTLZ3.. Otherwise, it takes 1.5
                tau = max(1.5 * tau, beta)
        else:
            self.logger.warn("Pre-conditioning the HV Hessian failed")
        return H - tau * I

    def _compute_G(self, X: np.ndarray) -> np.ndarray:
        # TODO: correct it for active set method
        # `_compute_G`` -> `_compute_R`
        N = len(X)
        mud = int(N * self.dim_primal)
        primal_vars, dual_vars = self._get_primal_dual(X)
        out = self.hypervolume_derivatives._compute_gradient(primal_vars)
        HVdX = out["HVdX"].ravel()

        dH = block_diag(*[self.h_jac(x) for x in primal_vars])
        eq_cstr = np.array([self.h(_) for _ in primal_vars]).reshape(N, -1)
        G = np.concatenate([HVdX + dual_vars.ravel() @ dH, eq_cstr.ravel()])
        return np.c_[G[:mud].reshape(N, -1), G[mud:].reshape(N, -1)]

    def _compute_netwon_step(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, np.ndarray]:
        N = X.shape[0]
        primal_vars, dual_vars = self._get_primal_dual(X)
        out = self.hypervolume_derivatives.compute(primal_vars, Y)
        HVdX, HVdX2 = out["HVdX"].ravel(), out["HVdX2"]
        # NOTE: preconditioning is needed EqDTLZ problems
        # if self.problem_name is not None and self.problem_name != "Eq1IDTLZ3":
        # HVdX2 = self._precondition_hessian(HVdX2)
        H, G = HVdX2, HVdX

        if self.h is not None:  # with equality constraints
            mud = int(N * self.dim_primal)
            mup = int(N * self.n_eq_cstr)
            # record the CPU time of function evaluations
            eq_cstr = np.array([self.h(_) for _ in primal_vars]).reshape(N, -1)  # (mu, p)
            dH = block_diag(*np.array([self.h_jac(x) for x in primal_vars]))  # (mu * p, mu * dim)
            ddH = block_diag(
                # NOTE: `np.einsum` is quite slow comparing the alternatives in np
                # TODO: ad-hoc solutions for now, which only works for one constraint.
                # Find a generic and faster solution later
                # *[np.einsum("ijk,i->jk", self._h_hessian(x), dual_vars[i]) for i, x in enumerate(primal_vars)]
                *[(self._h_hessian(x) * dual_vars[i])[0] for i, x in enumerate(primal_vars)]
            )  # (mu * dim, mu * dim)
            G = np.concatenate([HVdX + dual_vars.ravel() @ dH, eq_cstr.ravel()])
            # NOTE: if the Hessian of the constraint is dropped, then quadratic convergence is gone
            H = np.concatenate(
                [
                    np.concatenate([HVdX2 + ddH, dH.T], axis=1),
                    np.concatenate([dH, np.zeros((mup, mup))], axis=1),
                ],
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                # NOTE: use the sparse matrix representation to save some time here
                step = -1 * spsolve(csc_matrix(H), csc_matrix(G.reshape(-1, 1)))
            except:
                # NOTE: this part should not occur
                w, V = np.linalg.eigh(H)
                w[np.isclose(w, 0)] = 1e-6
                D = np.diag(1 / w)
                step = -1 * V @ D @ V.T @ G

        if self.h is not None:
            step = np.c_[step[:mud].reshape(N, -1), step[mud:].reshape(N, -1)]
            G = np.c_[G[:mud].reshape(N, -1), G[mud:].reshape(N, -1)]
            return dict(step=step, G=G)
        else:
            return dict(step=step.reshape(N, -1), G=HVdX.reshape(N, -1))

    def one_step(self):
        self._check_XY()
        self.step = np.zeros((self.mu, self.dim))
        self.step_size = np.ones(self.mu)
        self.G = np.zeros((self.mu, self.dim))

        # partition the approximation set to by feasibility
        self._nondominated_idx = non_domin_sort(self.Y, only_front_indices=True)[0]
        if self.h is None:
            feasible_mask = np.array([True] * self.mu)
        else:
            eq_cstr = np.array([self.h(_) for _ in self._get_primal_dual(self.X)[0]]).reshape(self.mu, -1)
            feasible_mask = np.all(np.isclose(eq_cstr, 0, atol=1e-4, rtol=0), axis=1)

        feasible_idx = np.nonzero(feasible_mask)[0]
        dominated_idx = list((set(range(self.mu)) - set(self._nondominated_idx) - set(feasible_idx)))
        if np.any(feasible_mask):
            # non-dominatd sorting of the feasible points
            partitions = non_domin_sort(self.Y[feasible_mask], only_front_indices=True)
            partitions = {k: feasible_idx[v] for k, v in partitions.items()}
            partitions.update({0: np.sort(np.r_[partitions[0], np.nonzero(~feasible_mask)[0]])})
        else:
            partitions = {0: np.array(range(self.mu))}

        # compute the Newton direction for each partition
        for _, idx in partitions.items():
            out = self._compute_netwon_step(X=self.X[idx], Y=self.Y[idx])
            self.step[idx, :] = out["step"]
            self.G[idx, :] = out["G"]
            # backtracking line search with Armijo's condition for each layer
            if _ == 0 and len(dominated_idx) > 0:  # for the first layer
                idx_ = list(set(idx) - set(dominated_idx))
                for k in dominated_idx:  # for dominated and infeasible points
                    self.step_size[k] = self._line_search_dominated(self.X[[k]], self.step[[k]])
                self.step_size[idx_] = self._line_search(self.X[idx_], self.step[idx_], G=self.G[idx_])
            else:  # for all other layers
                self.step_size[idx] = self._line_search(self.X[idx], self.step[idx], G=self.G[idx])

        self.X += self.step_size.reshape(-1, 1) * self.step
        # evaluation
        self.Y = np.array([self.func(x) for x in self._get_primal_dual(self.X)[0]])
        self.iter_count += 1

    def _line_search(self, X: np.ndarray, step: np.ndarray, G: np.ndarray) -> float:
        """backtracking line search with Armijo's condition"""
        # TODO: ad-hoc! to solve this in the further using a high precision numerical library
        # NOTE: when the step length is close to numpy's numerical resolution, it makes no sense to perform
        # the step-size control
        if np.any(np.isclose(np.median(step), np.finfo(np.double).resolution)):
            return 1

        c = 1e-5
        N = len(X)
        primal_vars = self._get_primal_dual(X)[0]
        normal_vectors = np.c_[np.eye(self.dim_primal * N), -1 * np.eye(self.dim_primal * N)]
        # calculate the maximal step-size
        dist = np.r_[
            np.abs(primal_vars.ravel() - np.tile(self.lower_bounds, N)),
            np.abs(np.tile(self.upper_bounds, N) - primal_vars.ravel()),
        ]
        v = step[:, : self.dim_primal].ravel() @ normal_vectors
        alpha = min(1, 0.25 * np.min(dist[v < 0] / np.abs(v[v < 0])))

        for _ in range(6):
            X_ = X + alpha * step
            if self.h is None:
                # TODO: check if this is correct
                HV = self.hypervolume_derivatives.HV(X)
                HV_ = self.hypervolume_derivatives.HV(X_)
                inc = np.inner(G.ravel(), step.ravel())
                cond = HV_ - HV >= c * alpha * inc
            else:
                G_ = self._compute_G(X_)
                cond = np.linalg.norm(G_) <= (1 - c * alpha) * np.linalg.norm(G)
            if cond:
                break
            else:
                if 11 < 2:
                    phi0 = HV if self.h is None else np.sum(G**2) / 2
                    phi1 = HV_ if self.h is None else np.sum(G_**2) / 2
                    phi0prime = inc if self.h is None else -np.sum(G**2)
                    alpha = -phi0prime * alpha**2 / (phi1 - phi0 - phi0prime * alpha) / 2
                    # alpha *= tau
                if 1 < 2:
                    alpha *= 0.5
        else:
            self.logger.warn("Armijo's backtracking line search failed")
        return alpha

    def _line_search_dominated(self, X: np.ndarray, step: np.ndarray) -> float:
        """backtracking line search with Armijo's condition"""
        # TODO: ad-hoc! to solve this in the further using a high precision numerical library
        # NOTE: when the step length is close to numpy's numerical resolution, it makes no sense to perform
        # the step-size control
        if np.any(np.isclose(np.median(step), np.finfo(np.double).resolution)):
            return 1

        c = 1e-4
        N = len(X)
        step = step[:, : self.dim_primal]
        primal_vars = self._get_primal_dual(X)[0]
        normal_vectors = np.c_[np.eye(self.dim_primal * N), -1 * np.eye(self.dim_primal * N)]
        # calculate the maximal step-size
        dist = np.r_[
            np.abs(primal_vars.ravel() - np.tile(self.lower_bounds, N)),
            np.abs(np.tile(self.upper_bounds, N) - primal_vars.ravel()),
        ]
        v = step.ravel() @ normal_vectors
        alpha = min(1, 0.25 * np.min(dist[v < 0] / np.abs(v[v < 0])))

        h_ = self.h(primal_vars)
        eq_cstr = h_**2 / 2
        G = h_ * self.h_jac(primal_vars)
        for _ in range(6):
            X_ = primal_vars + alpha * step
            eq_cstr_ = self.h(X_) ** 2 / 2
            dec = np.inner(G.ravel(), step.ravel())
            cond = eq_cstr_ - eq_cstr <= c * alpha * dec
            if cond:
                break
            else:
                alpha *= 0.5
        else:
            self.logger.warn("Armijo's backtracking line search failed")
        return alpha

    def _check_XY(self):
        # get unique points: if some points converge to the same location
        primal_vars = self.X[:, : self.dim_primal]
        D = cdist(primal_vars, primal_vars)
        drop_idx_X = set([])
        for i in range(self.mu):
            if i not in drop_idx_X:
                drop_idx_X |= set(np.nonzero(np.isclose(D[i, :], 0, rtol=self.eps))[0]) - set([i])

        # get rid of weakly-dominated points
        # TODO: Ad-hoc solution! check if this is still needed
        # since the hypervolume indicator module is upgraded
        drop_idx_Y = set([])
        # TODO: Ad-hoc solution! check if this is still needed
        if self.problem_name is not None and self.problem_name not in (
            "Eq1DTLZ4",
            "Eq1IDTLZ4",
        ):
            for i in range(self.mu):
                if i not in drop_idx_Y:
                    drop_idx_Y |= set(np.nonzero(np.any(np.isclose(self.Y[i, :], self.Y), axis=1))[0]) - set(
                        [i]
                    )
        idx = list(set(range(self.mu)) - (drop_idx_X | drop_idx_Y))
        self.mu = len(idx)
        self.X = self.X[idx, :]
        self.Y = self.Y[idx, :]

    def log(self):
        HV = hypervolume(self.Y, self.ref)
        self.hist_Y += [self.Y.copy()]
        self.hist_X += [self._get_primal_dual(self.X.copy())[0]]
        self.hist_HV += [HV]

        if self.verbose:
            self.logger.info(f"iteration {self.iter_count} ---")
            self.logger.info(f"HV: {HV}")
            # self.logger.info(f"step size: {self.step_size}")
            self.logger.info(f"#non-dominated: {len(self._nondominated_idx)}")

        if self.iter_count >= 1:
            try:
                self._delta_X = np.mean(np.sqrt(np.sum((self.hist_X[-1] - self.hist_X[-2]) ** 2, axis=1)))
                self._delta_Y = np.mean(np.sqrt(np.sum((self.hist_Y[-1] - self.hist_Y[-2]) ** 2, axis=1)))
                self._delta_HV = np.abs(self.hist_HV[-1] - self.hist_HV[-2])
            except:
                pass

        if self.h is not None:
            self.hist_G_norm += [np.median(np.linalg.norm(self.G[self._nondominated_idx], axis=1))]
            self.logger.info(f"G norm: {self.hist_G_norm[-1]}")

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


class State:
    def __init__(
        self,
        dim_p: int,
        n_eq: int,
        n_ieq: int,
        func: callable,
        jac: callable,
        h: callable = None,
        h_jac: callable = None,
        g: callable = None,
        g_jac: callable = None,
    ) -> None:
        self.dim_p = dim_p
        self.n_eq = n_eq
        self.n_ieq = n_ieq
        self.func = func
        self.jac = jac
        self.h = h
        self.h_jac = h_jac
        self.g = g
        self.g_jac = g_jac
        self._constrained = self.g is not None or self.h is not None

    def update(self, X: np.ndarray, compute_gradient: bool = True):
        self.X = X
        primal_vars = self.primal
        self.Y = np.array([self.func(x) for x in primal_vars])
        self.J = np.array([self.jac(x) for x in primal_vars]) if compute_gradient else None
        if self._constrained:
            eq = self._evaluate_constraints(primal_vars, type="eq", compute_gradient=compute_gradient)
            ieq = self._evaluate_constraints(primal_vars, type="ieq", compute_gradient=compute_gradient)
            cstr_value, active_indices, dH = merge_lists(eq, ieq)
            self.cstr_value = cstr_value
            self.active_indices = active_indices
            self.dH = dH

    def update_one(self, x: np.ndarray, k: int):
        self.X[k] = x
        x = np.atleast_2d(x)
        primal_vars = x[:, : self.dim_p]
        self.Y[k] = self.func(primal_vars[0])
        self.J[k] = self.jac(primal_vars[0])
        if self._constrained:
            eq = self._evaluate_constraints(primal_vars, type="eq", compute_gradient=True)
            ieq = self._evaluate_constraints(primal_vars, type="ieq", compute_gradient=True)
            cstr_value, active_indices, dH = merge_lists(eq, ieq)
            self.cstr_value[k] = cstr_value
            self.active_indices[k] = active_indices
            self.dH[k] = dH

    @property
    def primal(self) -> np.ndarray:
        """Primal variables"""
        return self.X[:, : self.dim_p]

    @property
    def dual(self) -> np.ndarray:
        """Langranian dual variables"""
        return self.X[:, self.dim_p :]

    @property
    def H(self) -> np.ndarray:
        """Equality constraint values"""
        return self.cstr_value[:, : self.n_eq]

    @property
    def G(self) -> np.ndarray:
        """Inequality constraint values"""
        return self.cstr_value[:, self.n_eq :]

    def _evaluate_constraints(self, primal_vars: np.ndarray, type: str = "eq", compute_gradient: bool = True):
        N = len(primal_vars)
        func = self.h if type == "eq" else self.g
        jac = self.h_jac if type == "eq" else self.g_jac
        if func is None:
            return None
        value = np.array([func(x) for x in primal_vars]).reshape(N, -1)
        active_indices = [[True] * self.n_eq] * N if type == "eq" else [v >= -1e-3 for v in value]
        active_indices = np.array(active_indices)
        if compute_gradient:
            H = np.array([jac(x).reshape(-1, self.dim_p) for x in primal_vars])
        return (value, active_indices, H) if compute_gradient else (value, active_indices)


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
        self.state = State(self.dim_p, self.n_eq, self.n_ieq, func, jac, h, h_jac, g, g_jac)
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
            # NOTE: ad-hoc solution for CF2 problem since the Jacobian on the box boundary is not defined
            # X0 += 1e-5 * (X0 - self.xl == 0).astype(int)
            # X0 -= 1e-5 * (X0 - self.xu == 0).astype(int)
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
        # backtracking line search for the step size
        self.step_size = self._backtracking_line_search(self.step, self.R)
        # Newton iteration and evaluation
        self.state.update(self.state.X + self.step_size * self.step)

    def log(self):
        self.iter_count += 1
        self.hist_Y += [self.state.Y.copy()]
        self.hist_X += [self.state.primal.copy()]
        gd_value = self._perf_gd.compute(Y=self.state.Y)
        igd_value = self._perf_igd.compute(Y=self.state.Y)
        self.hist_GD += [gd_value]
        self.hist_IGD += [self.IGD_value]
        self.hist_R_norm += [np.median(np.linalg.norm(self.R, axis=1))]

        if self.iter_count >= 2:
            self._delta_X = np.mean(np.sqrt(np.sum((self.hist_X[-1] - self.hist_X[-2]) ** 2, axis=1)))
            self._delta_Y = np.mean(np.sqrt(np.sum((self.hist_Y[-1] - self.hist_Y[-2]) ** 2, axis=1)))
            self._delta_GD = np.abs(self.hist_GD[-1] - self.hist_GD[-2])
            self._delta_IGD = np.abs(self.hist_IGD[-1] - self.hist_IGD[-2])

        if self.verbose:
            self.logger.info(f"iteration {self.iter_count} ---")
            self.logger.info(f"GD/IGD: {self.GD_value, self.IGD_value}")
            self.logger.info(f"step size: {self.step_size.ravel()}")
            if self._constrained:
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
                primal_vars, state.Y, self.Y_label, compute_hessian=False, Jacobian=state.J
            )
        R = grad  # the unconstrained case
        dH, idx = np.array([]), None
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
        grad, Hessian = self.active_indicator.compute_derivatives(
            primal_vars, self.state.Y, self.Y_label, Jacobian=self.state.J
        )
        # the root-finding problem and the gradient of the active constraints
        R_list, dH, active_indices = self._compute_R(self.state, grad=grad)
        idx = np.array([[True] * self.dim_p] * self.N)
        if active_indices is not None:
            idx = np.c_[idx, active_indices]
        # compute the Newton step for each approximation point - lower computation costs
        for r in range(self.N):
            c, dh = idx[r], dH[r]
            Z = np.zeros((len(dh), len(dh)))
            DR = np.r_[np.c_[Hessian[r], dh.T], np.c_[dh, Z]] if self._constrained else Hessian[r]
            R[r, c] = R_list[r]
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    newton_step[r, c] = -1 * solve(DR, R_list[r].reshape(-1, 1)).ravel()
                except Exception:
                    newton_step[r, c] = (
                        -1 * np.linalg.lstsq(DR, R_list[r].reshape(-1, 1), rcond=None)[0].ravel()
                    )
        return newton_step, R

    def _shift_reference_set(self):
        """shift the reference set when the following conditions are True:
        1. always shift the first reference set (`self.iter_count == 0`); Otherwise, weird matching can happen
        2. if at least one approximation point is close to its matched target and the Newton step is not zero.
        """
        distance = np.linalg.norm(self.state.Y - self.active_indicator._medoids, axis=1)
        masks = (
            np.array([True] * self.N)
            if self.iter_count == 0
            else np.bitwise_and(
                np.isclose(distance, 0),
                np.isclose(np.linalg.norm(self.step[:, : self.dim_p], axis=1), 0),
            )
        )
        indices = np.nonzero(masks)[0]
        if len(indices) == 0:
            return

        if self._eta is None:
            self._eta = dict()
            for i in range(self.n_cluster):
                Y = self.state.Y[self.Y_idx[i]]
                idx = non_domin_sort(Y, only_front_indices=True)[0]
                self._eta[i] = compute_chim(Y[idx])

        # shift the medoids
        for i, k in enumerate(indices):
            n = self._eta[self.Y_label[k]]
            v = 0.05 * n if self.iter_count > 0 else 0.03 * n  # the initial shift is a bit larger
            self.active_indicator.shift_medoids(v, k)

        if self.iter_count == 0:  # record the initial medoids
            self.history_medoids = [[m.copy()] for m in self.active_indicator._medoids]
        else:
            # log the updated medoids
            for i, k in enumerate(indices):
                self.history_medoids[k].append(self.active_indicator._medoids[indices][i])
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
            self.active_indicator.re_match = False
            R_ = self._compute_R(state)[0][i]
            self.active_indicator.re_match = True
            return np.linalg.norm(R_)

        if max_step_size is not None:
            step_size = max_step_size.reshape(-1, 1)
        else:
            step_size = np.ones((self.N, 1))
        for i in range(self.N):
            phi = [np.linalg.norm(R[i])]
            s = [0, 1]
            for _ in range(10):
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

    def handle_box_constraint(self, step: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
