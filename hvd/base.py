import pickle
from collections.abc import Callable
from copy import deepcopy
from typing import Self

import numpy as np
from scipy.optimize import linprog

ArrayFunction = Callable[[np.ndarray], np.ndarray]


class State:
    def __init__(
        self,
        n_var: int,
        n_eq: int,
        n_ieq: int,
        func: ArrayFunction,
        jac: ArrayFunction,
        h: ArrayFunction | None = None,
        h_jac: ArrayFunction | None = None,
        h_hess: ArrayFunction | None = None,
        g: ArrayFunction | None = None,
        g_jac: ArrayFunction | None = None,
        g_hess: ArrayFunction | None = None,
    ) -> None:
        """State object of numerical optimization

        Args:
            n_var (int): number of decision variable
            n_eq (int): number of equality constraints
            n_ieq (int): number of inequality constraints
            func (Callable): objective function
            jac (Callable): Jacobian of the objective function
            h (Callable, optional): equality constraint. Defaults to None.
            h_jac (Callable, optional): Jacobian of the equality constraint. Defaults to None.
            h_hess (Callable, optional): Hessian of the equality constraint. Defaults to None.
            g (Callable, optional): inequality constraint. Defaults to None.
            g_jac (Callable, optional): Jacobian of the inequality constraint. Defaults to None.
            g_hess (Callable, optional): Hessian of the inequality constraint. Defaults to None.
        """
        self.n_var: int = n_var  # the number of the primal variables
        self.n_eq: int = n_eq
        self.n_ieq: int = n_ieq
        self.n_cstr: int = self.n_eq + self.n_ieq
        self.func: ArrayFunction = func
        self.jac: ArrayFunction = jac
        self.h: ArrayFunction | None = h
        self.h_jac: ArrayFunction | None = h_jac
        self.h_hess: ArrayFunction | None = h_hess
        self.g: ArrayFunction | None = g
        self.g_jac: ArrayFunction | None = g_jac
        self.g_hess: ArrayFunction | None = g_hess
        self._constrained: bool = self.g is not None or self.h is not None
        self.n_jac_evals: int = 0
        self.n_cstr_jac_evals: int = 0
        self.n_cstr_hess_evals: int = 0

    @property
    def N(self) -> int:
        return len(self.X)

    @staticmethod
    def _get_batch_variant(function: ArrayFunction) -> ArrayFunction:
        """Return the batch implementation associated with a callback.

        Native problem methods always expose ``<method>_batch``.  The small
        row-wise adapter preserves support for legacy standalone callbacks.
        """
        owner = getattr(function, "__self__", None)
        name = getattr(function, "__name__", None)
        batch_function = getattr(owner, f"{name}_batch", None) if owner is not None and name else None
        if callable(batch_function):
            return batch_function

        def evaluate_rows(x: np.ndarray) -> np.ndarray:
            return np.stack([function(row) for row in x])

        return evaluate_rows

    @classmethod
    def evaluate(
        cls,
        function: ArrayFunction | None,
        x: np.ndarray,
        output_shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Evaluate any callback at one point or a population.

        A one-dimensional input calls the point function; a two-dimensional
        input calls its batch variant.  ``output_shape`` describes one point's
        output and is used to keep empty constraints and scalar outputs
        shape-stable.
        """
        x = np.asarray(x)
        if x.ndim not in (1, 2):
            raise ValueError("Evaluation input must have shape `(n_var,)` or `(n_points, n_var)`.")

        prefix = (len(x),) if x.ndim == 2 else ()
        if function is None:
            if output_shape is None:
                raise ValueError("`output_shape` is required when the function is None.")
            return np.zeros((*prefix, *output_shape))

        evaluator = function if x.ndim == 1 else cls._get_batch_variant(function)
        values = np.asarray(evaluator(x))
        return values if output_shape is None else values.reshape(*prefix, *output_shape)

    @staticmethod
    def _count(x: np.ndarray) -> int:
        return len(x) if np.ndim(x) == 2 else 1

    def eval_jac(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of the objective function"""
        self.n_jac_evals += self._count(x)
        return self.evaluate(self.jac, x)

    def eval_cstr(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x)
        H = self.evaluate(self.h, x, (self.n_eq,))
        G = self.evaluate(self.g, x, (self.n_ieq,))
        axis = 1 if x.ndim == 2 else 0
        values = np.concatenate((H, G), axis=axis)
        equality_active = np.ones((*values.shape[:-1], self.n_eq), dtype=bool)
        active_indices = np.concatenate((equality_active, G >= -1e-4), axis=axis)
        return values, active_indices

    def eval_cstr_jac(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        prefix = (len(x),) if x.ndim == 2 else ()
        if self.h_jac is None and self.g_jac is None:
            return np.empty((*prefix, 0, self.n_var))

        self.n_cstr_jac_evals += self._count(x)
        dH = self.evaluate(self.h_jac, x, (self.n_eq, self.n_var))
        dG = self.evaluate(self.g_jac, x, (self.n_ieq, self.n_var))
        return np.concatenate((dH, dG), axis=-2)

    def eval_cstr_hess(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        prefix = (len(x),) if x.ndim == 2 else ()
        if self.h_hess is None and self.g_hess is None:
            return np.zeros((*prefix, self.n_cstr, self.n_var, self.n_var))

        self.n_cstr_hess_evals += self._count(x)
        ddH = self.evaluate(self.h_hess, x, (self.n_eq, self.n_var, self.n_var))
        ddG = self.evaluate(self.g_hess, x, (self.n_ieq, self.n_var, self.n_var))
        return np.concatenate((ddH, ddG), axis=-3)

    def update(
        self, X: np.ndarray, compute_gradient: bool = True, compute_hessian: bool = True
    ) -> None:
        self.X = np.asarray(X).copy()
        self.Y = self.evaluate(self.func, self.primal)
        self.J = self.eval_jac(self.primal) if compute_gradient else None
        self.cstr_value, self.active_indices = self.eval_cstr(self.primal)
        self.cstr_grad = self.eval_cstr_jac(self.primal)
        if compute_hessian:
            self.cstr_hess = self.eval_cstr_hess(self.primal)

    def update_one(self, x: np.ndarray, k: int) -> None:
        primal_vars = x[: self.n_var]
        self.X[k] = x
        self.Y[k] = self.evaluate(self.func, primal_vars)
        self.J[k] = self.eval_jac(primal_vars)
        self.cstr_value[k], self.active_indices[k] = self.eval_cstr(primal_vars)
        self.cstr_grad[k] = self.eval_cstr_jac(primal_vars)
        self.cstr_hess[k] = self.eval_cstr_hess(primal_vars)

    def is_feasible(self) -> np.ndarray:
        """Check whether the solutions are feasible.
        NOTE: the active iequality constraints are considered infeasible

        Returns:
            np.ndarray: Boolean array of feasibility
        """
        ieq_active_idx = self.active_indices[:, self.n_eq :]
        eq_feasible = np.all(np.isclose(self.H, 0, atol=1e-4, rtol=0), axis=1)
        ieq_feasible = ~np.any(ieq_active_idx, axis=1)
        return np.bitwise_and(eq_feasible, ieq_feasible)

    def check_KKT(
        self,
        stationarity_tol: float = 1e-4,
        feasibility_tol: float = 1e-4,
        active_tol: float = 1e-4,
    ) -> np.ndarray:
        """Approximately test first-order multi-objective KKT conditions.

        Inequality constraints use the convention ``g(x) <= 0``.  For every
        feasible point, a small linear program minimizes the infinity norm of
        the stationarity residual subject to nonnegative objective weights
        summing to one, free equality multipliers, and nonnegative multipliers
        for active inequalities.
        """
        tolerances = (stationarity_tol, feasibility_tol, active_tol)
        if any(tolerance < 0 for tolerance in tolerances):
            raise ValueError("KKT tolerances must be non-negative.")
        if self.J is None:
            raise RuntimeError("Objective Jacobians are required to check KKT conditions.")
        if self.n_cstr and self.cstr_grad.shape[1] != self.n_cstr:
            raise RuntimeError("Constraint Jacobians are required to check KKT conditions.")

        is_KKT = np.zeros(self.N, dtype=np.bool_)
        for i in range(self.N):
            H, G = self.H[i], self.G[i]
            if (
                not np.all(np.isfinite(H))
                or not np.all(np.isfinite(G))
                or np.any(np.abs(H) > feasibility_tol)
                or np.any(G > feasibility_tol)
            ):
                continue

            objective_jacobian = np.asarray(self.J[i]).reshape(-1, self.n_var)
            n_obj = len(objective_jacobian)
            equality_jacobian = self.cstr_grad[i, : self.n_eq].reshape(self.n_eq, self.n_var)
            active_ieq = G >= -active_tol
            inequality_jacobian = self.cstr_grad[i, self.n_eq :][active_ieq].reshape(
                np.count_nonzero(active_ieq), self.n_var
            )
            stationarity_matrix = np.concatenate(
                (
                    objective_jacobian.T,
                    equality_jacobian.T,
                    inequality_jacobian.T,
                ),
                axis=1,
            )
            if not np.all(np.isfinite(stationarity_matrix)):
                continue

            n_multipliers = stationarity_matrix.shape[1]
            objective = np.zeros(n_multipliers + 1)
            objective[-1] = 1.0
            residual_bound = np.c_[
                np.vstack((stationarity_matrix, -stationarity_matrix)),
                -np.ones(2 * self.n_var),
            ]
            normalization = np.zeros((1, n_multipliers + 1))
            normalization[0, :n_obj] = 1.0
            bounds = (
                [(0.0, None)] * n_obj
                + [(None, None)] * self.n_eq
                + [(0.0, None)] * np.count_nonzero(active_ieq)
                + [(0.0, None)]
            )
            result = linprog(
                objective,
                A_ub=residual_bound,
                b_ub=np.zeros(2 * self.n_var),
                A_eq=normalization,
                b_eq=np.ones(1),
                bounds=bounds,
                method="highs",
            )
            is_KKT[i] = bool(result.success and result.x[-1] <= stationarity_tol)
        return is_KKT

    @property
    def primal(self) -> np.ndarray:
        """Primal variables"""
        return self.X[:, : self.n_var]

    @property
    def dual(self) -> np.ndarray:
        """Langranian dual variables"""
        return self.X[:, self.n_var :]

    @property
    def H(self) -> np.ndarray:
        """Equality constraint values"""
        return self.cstr_value[:, : self.n_eq]

    @property
    def G(self) -> np.ndarray:
        """Inequality constraint values"""
        return self.cstr_value[:, self.n_eq :]

    def save(self, filename: str) -> None:
        with open(filename + ".pkl", "wb") as file:
            __dict__ = self.__dict__.copy()
            del __dict__["func"]
            del __dict__["_jac"]
            del __dict__["h"]
            del __dict__["h_jac"]
            del __dict__["h_hess"]
            del __dict__["g"]
            del __dict__["g_jac"]
            del __dict__["g_hess"]
            pickle.dump(__dict__, file)

    @classmethod
    def load(cls, filename: str) -> Self:
        with open(filename, "rb") as file:
            data = pickle.load(file)
            obj = cls.__new__(cls)
            obj.__dict__.update(data)
            return obj

    def __getitem__(self, indices: np.ndarray) -> Self:
        """Slicing the state class

        Args:
            indices (np.ndarray): indices to select
        """
        obj = deepcopy(self)
        obj.n_jac_evals = obj.n_jac_evals
        obj.X = obj.X[indices]
        obj.Y = obj.Y[indices]
        obj.J = obj.J[indices]
        obj.cstr_value = obj.cstr_value[indices]
        obj.active_indices = obj.active_indices[indices]
        obj.cstr_grad = obj.cstr_grad[indices]
        obj.cstr_hess = obj.cstr_hess[indices]
        return obj
