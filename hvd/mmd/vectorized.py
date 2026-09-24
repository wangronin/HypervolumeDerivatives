from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Callable, Dict, Iterator, List, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import jit

from ..reference_set import ReferenceSet
from .kernels import RBF, CallableKernel, Kernel, KernelFunction


@jit
def _assemble_decision_hessian(MMDdY2, MMDdY, YdX, YdX2):
    """Apply the second-order chain rule to all point-pair blocks."""
    blocks = jnp.einsum("mai,mnab,nbj->mnij", YdX, MMDdY2, YdX)
    diagonal = jnp.einsum("ma,maij->mij", MMDdY, YdX2)
    indices = jnp.arange(len(YdX))
    blocks = blocks.at[indices, indices].add(diagonal)
    return blocks.transpose(0, 2, 1, 3).reshape(len(YdX) * YdX.shape[2], len(YdX) * YdX.shape[2])


class _VectorizedMMDBase:
    def __init__(
        self,
        n_var: int,
        n_obj: int,
        ref: Union[np.ndarray, ReferenceSet],
        func: Callable = None,
        jac: Callable = None,
        hessian: Callable = None,
        kernel: KernelFunction = RBF(),
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
                f"reference set must have shape (n_points, {self.n_obj}), " f"got {reference_set.shape}"
            )
        if len(reference_set) == 0:
            raise ValueError("reference set must contain at least one point")

        self.func = func if func is not None else lambda x: x
        self.jac = jac if jac is not None else lambda x: np.eye(self.n_obj, self.n_var)
        self.hessian = (
            hessian if hessian is not None else lambda x: np.zeros((self.n_obj, self.n_var, self.n_var))
        )
        self.n_decision_var = self.n_var
        self.n_objective = self.n_obj
        self.ref = ref
        self.N = self.ref.N
        self.kernel = kernel if isinstance(kernel, Kernel) else CallableKernel(kernel)
        self.history_reference_set: Dict[int, List[np.ndarray]] = defaultdict(list)

    def trial_evaluation(self) -> AbstractContextManager[None]:
        """Keep the indicator's evaluation state fixed during a trial step.

        Ordinary MMD has no temporary evaluation state. Stateful indicators
        override this context to preserve the model used for the Newton step.
        """
        return nullcontext()

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

    def shift_reference_set(self, c: float = 0.04, indices: np.ndarray | None = None) -> None:
        """Shift the selected reference points and record their new positions.

        MMD selects original reference points; MMDMatching selects matched medoids.
        None shifts all current targets; an empty selection leaves them unchanged.
        The scale c multiplies each target's component shift direction.
        """
        if indices is None:
            indices = np.arange(self.ref.N)
        self.ref.shift(c, indices)
        self._record_reference_shift(indices)

    def _record_reference_shift(self, indices: np.ndarray) -> None:
        """Save copies of shifted targets so later shifts cannot change their history."""
        reference_set = self.ref.reference_set
        for k in indices:
            self.history_reference_set[k].append(reference_set[k].copy())

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
            raise ValueError(f"objective Jacobian must have shape {expected_jacobian_shape}, got {YdX.shape}")
        YdX2 = np.asarray([self.hessian(x) for x in X]) if compute_hessian else None
        expected_hessian_shape = (len(X), self.n_obj, self.n_var, self.n_var)
        if compute_hessian and YdX2.shape != expected_hessian_shape:
            raise ValueError(f"objective Hessian must have shape {expected_hessian_shape}, got {YdX2.shape}")
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
            self.kernel.pairwise(reference_set, reference_set).mean()
            + self.kernel.pairwise(Y, Y).mean()
            - 2 * self.kernel.pairwise(Y, reference_set).mean()
        )
        return float(value)

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

    def _objective_gradient(self, Y, reference_set):
        N, M = len(Y), len(reference_set)
        indices = jnp.arange(N)
        yy_dx = self.kernel.pairwise_gradient(Y, Y)
        yr_dx = self.kernel.pairwise_gradient(Y, reference_set)
        nonself_dx = yy_dx.sum(axis=1) - yy_dx[indices, indices]
        return (
            2 * nonself_dx / N**2
            + self.kernel.diagonal_gradient_batch(Y) / N**2
            - 2 * yr_dx.sum(axis=1) / (N * M)
        )

    def _objective_hessian(self, Y, reference_set):
        N, M = len(Y), len(reference_set)
        indices = jnp.arange(N)
        yy_dx2 = self.kernel.pairwise_hessian(Y, Y)
        yr_dx2 = self.kernel.pairwise_hessian(Y, reference_set)
        diagonal = (
            2 * (yy_dx2.sum(axis=1) - yy_dx2[indices, indices]) / N**2
            + self.kernel.diagonal_hessian_batch(Y) / N**2
            - 2 * yr_dx2.sum(axis=1) / (N * M)
        )
        blocks = 2 * self.kernel.pairwise_mixed_hessian(Y, Y) / N**2
        return blocks.at[indices, indices].set(diagonal)


class MMDMatching(_VectorizedMMDBase):
    """Vectorized matched MMD surrogate with a general RKHS distance term."""

    def __init__(self, *args, beta: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beta = float(beta)

    @property
    def re_match(self) -> bool:
        """Whether evaluations recompute an existing point-to-reference matching."""
        return self.ref.re_match

    @re_match.setter
    def re_match(self, value: bool) -> None:
        self.ref.re_match = value

    @contextmanager
    def trial_evaluation(self) -> Iterator[None]:
        """Reuse the current point matching while evaluating a trial step.

        A line search must evaluate the surrogate used for the Newton step.
        Always re-enable matching on exit, including when evaluation raises.
        """
        self.re_match = False
        try:
            yield
        finally:
            self.re_match = True

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
            self.kernel.diagonal_batch(Y)
            + self.kernel.diagonal_batch(matched_reference_set)
            - 2 * self.kernel.matched(Y, matched_reference_set)
        )
        value = self.beta * self.kernel.pairwise(Y, Y).mean() + squared_rkhs_distance.mean()
        return float(value)

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

    def _objective_gradient(self, Y, matched_reference_set):
        N = len(Y)
        indices = jnp.arange(N)
        yy_dx = self.kernel.pairwise_gradient(Y, Y)
        nonself_dx = yy_dx.sum(axis=1) - yy_dx[indices, indices]
        diagonal_dx = self.kernel.diagonal_gradient_batch(Y)
        spread = self.beta * (2 * nonself_dx + diagonal_dx) / N**2
        attraction = (diagonal_dx - 2 * self.kernel.matched_gradient(Y, matched_reference_set)) / N
        return spread + attraction

    def _objective_hessian(self, Y, matched_reference_set):
        N = len(Y)
        indices = jnp.arange(N)
        yy_dx2 = self.kernel.pairwise_hessian(Y, Y)
        diagonal_dx2 = self.kernel.diagonal_hessian_batch(Y)
        diagonal = self.beta * (2 * (yy_dx2.sum(axis=1) - yy_dx2[indices, indices]) + diagonal_dx2) / N**2
        diagonal += (diagonal_dx2 - 2 * self.kernel.matched_hessian(Y, matched_reference_set)) / N
        blocks = 2 * self.beta * self.kernel.pairwise_mixed_hessian(Y, Y) / N**2
        return blocks.at[indices, indices].set(diagonal)
