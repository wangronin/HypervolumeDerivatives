"""Callable kernels with immutable hyperparameters.

The existing ``theta`` parameterization is preserved: it multiplies squared
distance in RBF and rational quadratic kernels, and L1 distance in Laplace.
Create a new kernel to change its parameters so JIT-compiled derivatives
cannot retain stale values.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import jax.numpy as jnp
from jax import jacfwd, jacrev, jit, vmap

KernelFunction = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


class Kernel(ABC):
    """A kernel, its derivatives, and their population evaluations.

    Functions are compiled lazily and reused. Subclasses may override the
    derivatives with analytical implementations without changing MMD.
    Batch inputs X and Y contain one d-dimensional point per row.
    """

    @abstractmethod
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the kernel at two points."""

    @cached_property
    def gradient(self) -> KernelFunction:
        """Gradient with respect to the first argument."""
        return jit(jacrev(self))

    @cached_property
    def hessian(self) -> KernelFunction:
        """Hessian with respect to the first argument."""
        return jit(jacfwd(self.gradient))

    @cached_property
    def mixed_hessian(self) -> KernelFunction:
        """Derivative of the first-argument gradient with respect to the second."""
        return jit(jacfwd(self.gradient, argnums=1))

    @cached_property
    def diagonal(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The one-argument function k(x, x)."""
        return jit(lambda x: self(x, x))

    @cached_property
    def diagonal_gradient(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Total gradient of k(x, x), allowing both arguments to vary."""
        return jit(jacrev(self.diagonal))

    @cached_property
    def diagonal_hessian(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Total Hessian of k(x, x)."""
        return jit(jacfwd(self.diagonal_gradient))

    # All-pairs evaluation forms the Cartesian product of two populations.
    # The first two output axes always index rows of X and Y, respectively.
    @cached_property
    def pairwise(self) -> KernelFunction:
        """Return k(X[i], Y[j]) for all pairs, with shape (N, M).

        X has shape (N, d) and Y has shape (M, d); their sizes may differ.
        This is the kernel matrix used in MMD's within- and between-set sums.
        """
        return jit(vmap(vmap(self, in_axes=(None, 0)), in_axes=(0, None)))

    @cached_property
    def pairwise_gradient(self) -> KernelFunction:
        """Return first-argument gradients for all pairs, shaped (N, M, d).

        Entry [i, j, a] differentiates k(X[i], Y[j]) with respect to X[i, a].
        """
        return jit(vmap(vmap(self.gradient, in_axes=(None, 0)), in_axes=(0, None)))

    @cached_property
    def pairwise_hessian(self) -> KernelFunction:
        """Return first-argument Hessians for all pairs, shaped (N, M, d, d).

        Entry [i, j, a, b] differentiates twice with respect to X[i],
        in coordinates a and b, holding Y[j] fixed.
        """
        return jit(vmap(vmap(self.hessian, in_axes=(None, 0)), in_axes=(0, None)))

    @cached_property
    def pairwise_mixed_hessian(self) -> KernelFunction:
        """Return mixed Hessians for all pairs, shaped (N, M, d, d).

        Entry [i, j, a, b] differentiates with respect to X[i, a] and Y[j, b].
        MMD uses these blocks to couple distinct approximation points.
        """
        return jit(vmap(vmap(self.mixed_hessian, in_axes=(None, 0)), in_axes=(0, None)))

    # Self-pairs use the one-argument function k(x, x). Its total derivatives
    # include both arguments, unlike a diagonal slice of pairwise_gradient.
    @cached_property
    def diagonal_batch(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return k(X[i], X[i]) for X shaped (N, d), producing shape (N,).

        Evaluate only self-pairs, without constructing an N-by-N matrix.
        """
        return jit(vmap(self.diagonal))

    @cached_property
    def diagonal_gradient_batch(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return total gradients of k(X[i], X[i]), shaped (N, d).

        These supply the self-pair terms in the indicator gradient.
        """
        return jit(vmap(self.diagonal_gradient))

    @cached_property
    def diagonal_hessian_batch(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Return total Hessians of k(X[i], X[i]), shaped (N, d, d).

        These supply the self-pair terms in the indicator Hessian.
        """
        return jit(vmap(self.diagonal_hessian))

    # Matched evaluation consumes already aligned rows. Finding the matching
    # remains the reference set's responsibility; these functions never reorder.
    @cached_property
    def matched(self) -> KernelFunction:
        """Return k(X[i], Y[i]) for two (N, d) inputs, producing shape (N,).

        Evaluate only the N supplied pairs, without an all-pairs matrix.
        """
        return jit(vmap(self))

    @cached_property
    def matched_gradient(self) -> KernelFunction:
        """Return first-argument gradients for aligned pairs, shaped (N, d).

        Y[i] stays fixed while X[i] varies, as in MMD-Matching's attraction.
        """
        return jit(vmap(self.gradient))

    @cached_property
    def matched_hessian(self) -> KernelFunction:
        """Return first-argument Hessians for aligned pairs, shaped (N, d, d).

        Each block differentiates k(X[i], Y[i]) twice with Y[i] fixed.
        """
        return jit(vmap(self.hessian))

    def __getstate__(self) -> dict[str, object]:
        """Recreate cached JAX functions after serialization to worker processes."""
        state = self.__dict__.copy()
        for name in (
            "gradient",
            "hessian",
            "mixed_hessian",
            "diagonal",
            "diagonal_gradient",
            "diagonal_hessian",
            "pairwise",
            "pairwise_gradient",
            "pairwise_hessian",
            "pairwise_mixed_hessian",
            "diagonal_batch",
            "diagonal_gradient_batch",
            "diagonal_hessian_batch",
            "matched",
            "matched_gradient",
            "matched_hessian",
        ):
            state.pop(name, None)
        return state


@dataclass(frozen=True)
class RBF(Kernel):
    """Gaussian kernel: \exp(-\theta * \lVert x - y \rVert^2)."""

    theta: float = 1.0

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-self.theta * jnp.sum((x - y) ** 2))


@dataclass(frozen=True)
class RationalQuadratic(Kernel):
    theta: float = 1.0
    alpha: float = 1.0

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return (1 + self.theta * jnp.sum((x - y) ** 2) / (2 * self.alpha)) ** (-self.alpha)


@dataclass(frozen=True)
class Laplace(Kernel):
    """Exponential kernel based on L1 distance."""

    theta: float = 1.0

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-self.theta * jnp.sum(jnp.abs(x - y)))


@dataclass(frozen=True)
class CallableKernel(Kernel):
    """Give a custom JAX-compatible callable the kernel derivative interface."""

    function: KernelFunction

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return self.function(x, y)
