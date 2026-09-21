from functools import partial
from typing import Callable, Dict, List, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jit, vmap

from .mmd import laplace, linear, rational_quadratic, rbf
from .reference_set import ReferenceSet


@jit
def _assemble_decision_hessian(MMDdY2, MMDdY, YdX, YdX2):
    """Apply the second-order chain rule to all point-pair blocks."""
    blocks = jnp.einsum("mai,mnab,nbj->mnij", YdX, MMDdY2, YdX)
    diagonal = jnp.einsum("ma,maij->mij", MMDdY, YdX2)
    indices = jnp.arange(len(YdX))
    blocks = blocks.at[indices, indices].add(diagonal)
    return blocks.transpose(0, 2, 1, 3).reshape(
        len(YdX) * YdX.shape[2], len(YdX) * YdX.shape[2]
    )


class _VectorizedMMDBase:
    def __init__(
        self,
        n_var: int,
        n_obj: int,
        ref: Union[np.ndarray, ReferenceSet],
        func: Callable = None,
        jac: Callable = None,
        hessian: Callable = None,
        kernel: Callable = rbf,
        theta: float = 1.0,
    ) -> None:
        self.n_var = int(n_var)
        self.n_obj = int(n_obj)
        if self.n_var < 1 or self.n_obj < 1:
            raise ValueError("n_var and n_obj must be positive")
        if isinstance(ref, np.ndarray):
            ref = ReferenceSet(ref)
        reference_set = np.asarray(ref.reference_set)
        if reference_set.ndim != 2 or reference_set.shape[1] != self.n_obj:
            raise ValueError(
                f"reference set must have shape (n_points, {self.n_obj}), "
                f"got {reference_set.shape}"
            )
        if len(reference_set) == 0:
            raise ValueError("reference set must contain at least one point")
        if func is None and self.n_var != self.n_obj:
            raise ValueError("the default identity objective requires n_var == n_obj")
        self.func = func if func is not None else lambda x: x
        self.jac = jac if jac is not None else lambda x: np.eye(self.n_obj, self.n_var)
        self.hessian = (
            hessian
            if hessian is not None
            else lambda x: np.zeros((self.n_obj, self.n_var, self.n_var))
        )
        self.n_decision_var = self.n_var
        self.n_objective = self.n_obj
        self.theta = float(theta)
        self.ref = ref
        self.N = self.ref.N
        self.kernel = kernel
        self.k = partial(kernel, theta=self.theta)

        self.k_dx = jit(jacrev(self.k))
        self.k_dx2 = jit(jacfwd(jacrev(self.k)))
        self.k_dxdy = jit(jacfwd(jacrev(self.k), argnums=1))
        self.k_diag = jit(lambda x: self.k(x, x))
        self.k_diag_dx = jit(jacrev(self.k_diag))
        self.k_diag_dx2 = jit(jacfwd(jacrev(self.k_diag)))

        self._pairwise_k = jit(vmap(vmap(self.k, in_axes=(None, 0)), in_axes=(0, None)))
        self._pairwise_dx = jit(vmap(vmap(self.k_dx, in_axes=(None, 0)), in_axes=(0, None)))
        self._pairwise_dx2 = jit(vmap(vmap(self.k_dx2, in_axes=(None, 0)), in_axes=(0, None)))
        self._pairwise_dxdy = jit(vmap(vmap(self.k_dxdy, in_axes=(None, 0)), in_axes=(0, None)))
        self._diagonal_k = jit(vmap(self.k_diag))
        self._diagonal_dx = jit(vmap(self.k_diag_dx))
        self._diagonal_dx2 = jit(vmap(self.k_diag_dx2))
        self._matched_k = jit(vmap(self.k))
        self._matched_dx = jit(vmap(self.k_dx))
        self._matched_dx2 = jit(vmap(self.k_dx2))

    def compute_derivatives(
        self,
        X: np.ndarray,
        Y: np.ndarray = None,
        compute_hessian: bool = True,
        jacobian: np.ndarray = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if compute_hessian:
            out = self.compute_hessian(X, Y, jacobian)
            return out["MMDdX"], out["MMDdX2"]
        return self.compute_gradient(X, Y, jacobian)["MMDdX"]

    def _compute_objective_derivatives(
        self,
        X: np.ndarray,
        Y: np.ndarray = None,
        jacobian: np.ndarray = None,
        compute_hessian: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self._check_X(X)
        if Y is None:
            Y = np.asarray([self.func(x) for x in X])
        Y = self._check_Y(Y, n_points=len(X))
        YdX = np.asarray([self.jac(x) for x in X]) if jacobian is None else np.asarray(jacobian)
        expected_jacobian_shape = (len(X), self.n_obj, self.n_var)
        if YdX.shape != expected_jacobian_shape:
            raise ValueError(
                f"objective Jacobian must have shape {expected_jacobian_shape}, got {YdX.shape}"
            )
        YdX2 = np.asarray([self.hessian(x) for x in X]) if compute_hessian else None
        expected_hessian_shape = (len(X), self.n_obj, self.n_var, self.n_var)
        if compute_hessian and YdX2.shape != expected_hessian_shape:
            raise ValueError(
                f"objective Hessian must have shape {expected_hessian_shape}, got {YdX2.shape}"
            )
        return Y, YdX, YdX2

    def _decision_derivatives(self, MMDdY, MMDdY2_blocks, YdX, YdX2):
        MMDdX = np.einsum("ma,mai->mi", MMDdY, YdX)
        MMDdX2 = np.asarray(
            _assemble_decision_hessian(
                jnp.asarray(MMDdY2_blocks),
                jnp.asarray(MMDdY),
                jnp.asarray(YdX),
                jnp.asarray(YdX2),
            )
        )
        return MMDdX, MMDdX2

    def _check_X(self, X: Union[np.ndarray, List]) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if X.shape[1] != self.n_var:
            X = X.T
        if X.shape[1] != self.n_var:
            raise ValueError(f"expected {self.n_var} decision variables, got shape {X.shape}")
        if len(X) == 0:
            raise ValueError("X must contain at least one point")
        return X

    def _check_Y(self, Y: np.ndarray, n_points: int = None) -> np.ndarray:
        Y = np.asarray(Y)
        expected_rows = len(Y) if n_points is None else n_points
        expected_shape = (expected_rows, self.n_obj)
        if Y.ndim != 2 or Y.shape != expected_shape:
            raise ValueError(f"Y must have shape {expected_shape}, got {Y.shape}")
        if len(Y) == 0:
            raise ValueError("Y must contain at least one point")
        return Y

    @property
    def re_match(self) -> bool:
        """Expose the matching switch expected by ``MMDNewton``."""
        return self.ref.re_match

    @re_match.setter
    def re_match(self, value: bool) -> None:
        self.ref.re_match = value


class MMD(_VectorizedMMDBase):
    """Vectorized biased squared maximum mean discrepancy."""

    def compute(self, X: np.ndarray = None, Y: np.ndarray = None) -> float:
        if Y is None:
            if X is None:
                raise ValueError("either X or Y must be provided")
            Y = np.asarray([self.func(x) for x in self._check_X(X)])
        Y = jnp.asarray(self._check_Y(Y))
        reference_set = jnp.asarray(self.ref.reference_set)
        value = (
            self._pairwise_k(reference_set, reference_set).mean()
            + self._pairwise_k(Y, Y).mean()
            - 2 * self._pairwise_k(Y, reference_set).mean()
        )
        return float(value)

    def _objective_gradient(self, Y, reference_set):
        N, M = len(Y), len(reference_set)
        indices = jnp.arange(N)
        yy_dx = self._pairwise_dx(Y, Y)
        yr_dx = self._pairwise_dx(Y, reference_set)
        nonself_dx = yy_dx.sum(axis=1) - yy_dx[indices, indices]
        return (
            2 * nonself_dx / N**2
            + self._diagonal_dx(Y) / N**2
            - 2 * yr_dx.sum(axis=1) / (N * M)
        )

    def _objective_hessian(self, Y, reference_set):
        N, M = len(Y), len(reference_set)
        indices = jnp.arange(N)
        yy_dx2 = self._pairwise_dx2(Y, Y)
        yr_dx2 = self._pairwise_dx2(Y, reference_set)
        diagonal = (
            2 * (yy_dx2.sum(axis=1) - yy_dx2[indices, indices]) / N**2
            + self._diagonal_dx2(Y) / N**2
            - 2 * yr_dx2.sum(axis=1) / (N * M)
        )
        blocks = 2 * self._pairwise_dxdy(Y, Y) / N**2
        return blocks.at[indices, indices].set(diagonal)

    def compute_gradient(
        self, X: np.ndarray, Y: np.ndarray = None, jacobian: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        Y, YdX, _ = self._compute_objective_derivatives(X, Y, jacobian, compute_hessian=False)
        MMDdY = np.asarray(self._objective_gradient(jnp.asarray(Y), jnp.asarray(self.ref.reference_set)))
        MMDdX = np.einsum("ma,mai->mi", MMDdY, YdX)
        return dict(MMDdX=MMDdX, MMDdY=MMDdY, Y=Y, YdX=YdX, YdX2=None)

    def compute_hessian(
        self, X: np.ndarray, Y: np.ndarray = None, jacobian: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        Y, YdX, YdX2 = self._compute_objective_derivatives(X, Y, jacobian, compute_hessian=True)
        Y_jax = jnp.asarray(Y)
        reference_set = jnp.asarray(self.ref.reference_set)
        MMDdY = np.asarray(self._objective_gradient(Y_jax, reference_set))
        MMDdY2_blocks = np.asarray(self._objective_hessian(Y_jax, reference_set))
        MMDdX, MMDdX2 = self._decision_derivatives(MMDdY, MMDdY2_blocks, YdX, YdX2)
        N, dim_y = Y.shape
        MMDdY2 = MMDdY2_blocks.transpose(0, 2, 1, 3).reshape(N * dim_y, N * dim_y)
        return dict(
            MMDdX2=MMDdX2,
            MMDdY2=MMDdY2,
            MMDdX=MMDdX,
            MMDdY=MMDdY,
            Y=Y,
            YdX=YdX,
            YdX2=YdX2,
        )


class MMDMatching(_VectorizedMMDBase):
    """Vectorized matched MMD surrogate with a general RKHS distance term."""

    def __init__(self, *args, beta: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beta = float(beta)

    def _match(self, Y):
        self.ref.match(np.asarray(Y))
        return jnp.asarray(self.ref.reference_set)

    def compute(self, X: np.ndarray = None, Y: np.ndarray = None) -> float:
        if Y is None:
            if X is None:
                raise ValueError("either X or Y must be provided")
            Y = np.asarray([self.func(x) for x in self._check_X(X)])
        Y = jnp.asarray(self._check_Y(Y))
        matched_reference_set = self._match(Y)
        squared_rkhs_distance = (
            self._diagonal_k(Y)
            + self._diagonal_k(matched_reference_set)
            - 2 * self._matched_k(Y, matched_reference_set)
        )
        value = self.beta * self._pairwise_k(Y, Y).mean() + squared_rkhs_distance.mean()
        return float(value)

    def _objective_gradient(self, Y, matched_reference_set):
        N = len(Y)
        indices = jnp.arange(N)
        yy_dx = self._pairwise_dx(Y, Y)
        nonself_dx = yy_dx.sum(axis=1) - yy_dx[indices, indices]
        diagonal_dx = self._diagonal_dx(Y)
        spread = self.beta * (2 * nonself_dx + diagonal_dx) / N**2
        attraction = (diagonal_dx - 2 * self._matched_dx(Y, matched_reference_set)) / N
        return spread + attraction

    def _objective_hessian(self, Y, matched_reference_set):
        N = len(Y)
        indices = jnp.arange(N)
        yy_dx2 = self._pairwise_dx2(Y, Y)
        diagonal_dx2 = self._diagonal_dx2(Y)
        diagonal = self.beta * (
            2 * (yy_dx2.sum(axis=1) - yy_dx2[indices, indices]) + diagonal_dx2
        ) / N**2
        diagonal += (diagonal_dx2 - 2 * self._matched_dx2(Y, matched_reference_set)) / N
        blocks = 2 * self.beta * self._pairwise_dxdy(Y, Y) / N**2
        return blocks.at[indices, indices].set(diagonal)

    def compute_gradient(
        self, X: np.ndarray, Y: np.ndarray = None, jacobian: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        Y, YdX, _ = self._compute_objective_derivatives(X, Y, jacobian, compute_hessian=False)
        matched_reference_set = self._match(Y)
        MMDdY = np.asarray(self._objective_gradient(jnp.asarray(Y), matched_reference_set))
        MMDdX = np.einsum("ma,mai->mi", MMDdY, YdX)
        return dict(MMDdX=MMDdX, MMDdY=MMDdY, Y=Y, YdX=YdX, YdX2=None)

    def compute_hessian(
        self, X: np.ndarray, Y: np.ndarray = None, jacobian: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        Y, YdX, YdX2 = self._compute_objective_derivatives(X, Y, jacobian, compute_hessian=True)
        matched_reference_set = self._match(Y)
        Y_jax = jnp.asarray(Y)
        MMDdY = np.asarray(self._objective_gradient(Y_jax, matched_reference_set))
        MMDdY2_blocks = np.asarray(self._objective_hessian(Y_jax, matched_reference_set))
        MMDdX, MMDdX2 = self._decision_derivatives(MMDdY, MMDdY2_blocks, YdX, YdX2)
        N, dim_y = Y.shape
        MMDdY2 = MMDdY2_blocks.transpose(0, 2, 1, 3).reshape(N * dim_y, N * dim_y)
        return dict(
            MMDdX2=MMDdX2,
            MMDdY2=MMDdY2,
            MMDdX=MMDdX,
            MMDdY=MMDdY,
            Y=Y,
            YdX=YdX,
            YdX2=YdX2,
        )
