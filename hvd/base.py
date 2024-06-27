import numpy as np

from .utils import merge_lists


# TODO: add Hessian
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
        self._jac = jac
        self.h = h
        self.h_jac = h_jac
        self.g = g
        self.g_jac = g_jac
        self._constrained = self.g is not None or self.h is not None
        self.n_jac_evals = 0

    def jac(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of the objective function"""
        self.n_jac_evals += 1
        return self._jac(x)

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
        active_indices = [[True] * self.n_eq] * N if type == "eq" else [v >= -1e-4 for v in value]
        active_indices = np.array(active_indices)
        if compute_gradient:
            H = np.array([jac(x).reshape(-1, self.dim_p) for x in primal_vars])
        return (value, active_indices, H) if compute_gradient else (value, active_indices)
