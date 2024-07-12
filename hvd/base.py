from copy import deepcopy
from typing import Callable, Self, Tuple

import numpy as np


class State:
    def __init__(
        self,
        n_var: int,
        n_eq: int,
        n_ieq: int,
        func: Callable,
        jac: Callable,
        h: Callable = None,
        h_jac: Callable = None,
        h_hess: Callable = None,
        g: Callable = None,
        g_jac: Callable = None,
        g_hess: Callable = None,
    ) -> None:
        self.n_var: int = n_var  # the number of the primal variables
        self.n_eq: int = n_eq
        self.n_ieq: int = n_ieq
        self.n_cstr: int = self.n_eq + self.n_ieq
        self.func: Callable = func
        self._jac: Callable = jac
        self.h: Callable = h
        self.h_jac: Callable = h_jac
        self.h_hess: Callable = h_hess
        self.g: Callable = g
        self.g_jac: Callable = g_jac
        self.g_hess: Callable = g_hess
        self._constrained: bool = self.g is not None or self.h is not None
        self.n_jac_evals: int = 0
        self.n_cstr_jac_evals: int = 0
        self.n_cstr_hess_evals: int = 0

    @property
    def N(self):
        # TODO: maybe `N` -> `__len__`
        return len(self.X)

    def jac(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of the objective function"""
        self.n_jac_evals += 1
        return self._jac(x)

    def evaluate_cstr(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H = self.h(x) if self.h is not None else []
        G = self.g(x) if self.g is not None else []
        if isinstance(H, (int, float)):
            H = np.array([H])
        if isinstance(G, (int, float)):
            G = np.array([G])
        active_indices = [True] * self.n_eq
        if self.g is not None:
            active_indices += (G >= -1e-4).tolist()
        return np.r_[H, G], np.array(active_indices)

    def evaluate_cstr_jac(self, x: np.ndarray) -> np.ndarray:
        if self.g_jac is not None or self.h_jac is not None:
            self.n_cstr_jac_evals += 1
        # TODO: only evaluate active inequality constraints
        dH = self.h_jac(x).tolist() if self.h_jac is not None else []
        dG = self.g_jac(x).tolist() if self.g_jac is not None else []
        out = np.array(dH + dG)
        return out if len(out) == 0 else out.reshape(self.n_cstr, self.n_var)

    def evaluate_cstr_hess(self, x: np.ndarray) -> np.ndarray:
        if self.g_hess is not None or self.h_hess is not None:
            self.n_cstr_hess_evals += 1
        # TODO: only evaluate active inequality constraints
        ddH = self.h_hess(x).tolist() if self.h_hess is not None else []
        ddG = self.g_hess(x).tolist() if self.g_hess is not None else []
        out = np.array(ddH + ddG)
        return out if len(out) == 0 else out.reshape(self.n_cstr, self.n_var, self.n_var)

    def update(self, X: np.ndarray, compute_gradient: bool = True):
        self.X = X
        primal_vars = self.primal
        self.Y = np.array([self.func(x) for x in primal_vars])
        self.J = np.array([self.jac(x) for x in primal_vars]) if compute_gradient else None
        cstr_value, active_indices = list(zip(*[self.evaluate_cstr(x) for x in primal_vars]))
        self.cstr_value = np.array(cstr_value)
        self.active_indices = np.array(active_indices)
        self.cstr_grad = np.array([self.evaluate_cstr_jac(x) for x in primal_vars])
        self.cstr_hess = np.array([self.evaluate_cstr_hess(x) for x in primal_vars])

    def update_one(self, x: np.ndarray, k: int):
        primal_vars = x[: self.n_var]
        self.X[k] = x
        self.Y[k] = self.func(primal_vars)
        self.J[k] = self.jac(primal_vars)
        cstr_value, active_indices = self.evaluate_cstr(primal_vars)
        cstr_grad = self.evaluate_cstr_jac(primal_vars)
        cstr_hess = self.evaluate_cstr_hess(primal_vars)
        if cstr_value is not None:
            self.cstr_value[k] = cstr_value
            self.active_indices[k] = active_indices
        if cstr_grad is not None:
            self.cstr_grad[k] = cstr_grad
        if cstr_hess is not None:
            self.cstr_hess[k] = cstr_hess

    def is_feasible(self) -> np.ndarray:
        # TODO: should the active ieq cstr be considered infeasible?
        ieq_active_idx = self.active_indices[:, self.n_eq :]
        eq_feasible = np.all(np.isclose(self.H, 0, atol=1e-4, rtol=0), axis=1)
        ieq_feasible = ~np.any(ieq_active_idx, axis=1)
        return np.bitwise_and(eq_feasible, ieq_feasible)

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

    def __getitem__(self, indices: np.ndarray) -> Self:
        """Slicing the state class

        Args:
            indices (np.ndarray): indices to select
        """
        obj = deepcopy(self)
        obj.n_jac_evals = 0
        obj.X = obj.X[indices]
        obj.Y = obj.Y[indices]
        obj.J = obj.J[indices]
        if self._constrained:
            obj.cstr_value = obj.cstr_value[indices]
            obj.active_indices = obj.active_indices[indices]
            obj.cstr_grad = obj.cstr_grad[indices]
        return obj
