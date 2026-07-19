"""Factories for optional, JAX-backed pymoo problem adapters."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final

from .base import MOP

if TYPE_CHECKING:
    from pymoo.core.problem import Problem as PymooProblem

_ProblemSpec = tuple[str, str]

# Keep this registry explicit: inclusion means that the problem has been checked
# with this factory's JAX path and the MOP adapter.
_PYMOO_PROBLEMS: Final[dict[str, _ProblemSpec]] = {
    "bnh": ("pymoo.problems.multi.bnh", "BNH"),
    "osy": ("pymoo.problems.multi.osy", "OSY"),
    "srn": ("pymoo.problems.multi.srn", "SRN"),
    "tnk": ("pymoo.problems.multi.tnk", "TNK"),
    "zdt1": ("pymoo.problems.multi.zdt", "ZDT1"),
    "zdt2": ("pymoo.problems.multi.zdt", "ZDT2"),
    "zdt3": ("pymoo.problems.multi.zdt", "ZDT3"),
    "zdt4": ("pymoo.problems.multi.zdt", "ZDT4"),
    "zdt6": ("pymoo.problems.multi.zdt", "ZDT6"),
    "dtlz1": ("pymoo.problems.many.dtlz", "DTLZ1"),
    "dtlz2": ("pymoo.problems.many.dtlz", "DTLZ2"),
    "dtlz3": ("pymoo.problems.many.dtlz", "DTLZ3"),
    "dtlz4": ("pymoo.problems.many.dtlz", "DTLZ4"),
    "dtlz5": ("pymoo.problems.many.dtlz", "DTLZ5"),
    "dtlz6": ("pymoo.problems.many.dtlz", "DTLZ6"),
    "dtlz7": ("pymoo.problems.many.dtlz", "DTLZ7"),
}


def pymoo_problem_names() -> tuple[str, ...]:
    """Return the pymoo problem names supported by :func:`get_pymoo_problem`."""
    return tuple(_PYMOO_PROBLEMS)


@contextmanager
def _jax_pymoo_backend() -> Iterator[None]:
    """Bind modules imported in the context to JAX without changing pymoo globally.

    ``pymoo.gradient.activate`` changes which module a subsequent
    ``import pymoo.gradient.toolbox`` resolves to.  A problem module imported
    while JAX is active keeps its own ``anp`` reference to ``jax.numpy`` after
    this context exits.  Restoring the previous setting only controls future
    toolbox imports; it does not rewrite that already-imported problem module.
    """
    import pymoo.gradient as gradient

    previous_backend = gradient.TOOLBOX
    if previous_backend != "jax.numpy":
        gradient.activate("jax.numpy")
    try:
        yield
    finally:
        if previous_backend != "jax.numpy":
            gradient.activate(previous_backend)


def _import_jax_problem_type(module_name: str, class_name: str) -> "type[PymooProblem]":
    loaded_module = sys.modules.get(module_name)
    if loaded_module is not None:
        backend = getattr(getattr(loaded_module, "anp", None), "__name__", None)
        if backend != "jax.numpy":
            raise RuntimeError(
                f"{module_name!r} was imported with backend {backend!r} before the "
                "JAX-backed factory was called. Call `get_pymoo_problem` before "
                "importing `pymoo.problems`, or use a fresh Python process."
            )
        return getattr(loaded_module, class_name)

    with _jax_pymoo_backend():
        module = import_module(module_name)
        return getattr(module, class_name)


def get_pymoo_problem(
    name: str,
    *,
    boundary_constraints: bool = True,
    **problem_kwargs: Any,
) -> MOP:
    """Create a differentiable MOP from a selected built-in pymoo problem.

    The factory activates pymoo's JAX toolbox before importing the selected
    problem module and restores the previously configured toolbox afterward.
    Restoring the toolbox does not rebind modules that were imported while JAX
    was active. Because Python caches those modules, call this factory before
    importing the same ``multi`` or ``many`` family from :mod:`pymoo.problems`.
    """
    key = name.lower()
    try:
        module_name, class_name = _PYMOO_PROBLEMS[key]
    except KeyError as error:
        supported = ", ".join(pymoo_problem_names())
        raise ValueError(f"Unsupported pymoo problem {name!r}. Available problems: {supported}.") from error

    problem_type = _import_jax_problem_type(module_name, class_name)
    pymoo_problem = problem_type(**problem_kwargs)

    return MOP.from_pymoo(
        pymoo_problem,
        boundary_constraints=boundary_constraints,
    )
