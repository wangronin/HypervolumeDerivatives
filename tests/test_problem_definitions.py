import inspect

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import ArrayLike

from hvd.problems import CF1, CMOP, DTLZ1, MOP, UF1, UF8, ZDT1
from hvd.problems.misc import CONV3, DENT


class _ToyMOP(MOP):
    def __init__(
        self,
        n_var: int = 2,
        n_obj: int = 2,
        xl: ArrayLike = 0.0,
        xu: ArrayLike = 1.0,
    ) -> None:
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([jnp.sum(x), jnp.sum(x**2)])


class _ToyCMOP(CMOP):
    def __init__(
        self,
        n_var: int = 2,
        n_obj: int = 2,
        xl: ArrayLike = 0.0,
        xu: ArrayLike = 1.0,
        boundary_constraints: bool = False,
    ) -> None:
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
            n_eq_constr=1,
            n_ieq_constr=1,
            boundary_constraints=boundary_constraints,
        )

    def _objective(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([jnp.sum(x), jnp.sum(x**2)])

    def _eq_constraint(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([x[0] - x[1]])

    def _ieq_constraint(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([jnp.sum(x) - 0.5])


@pytest.mark.parametrize(
    ("problem_type", "expected_n_obj"),
    [(UF1, 2), (UF8, 3), (ZDT1, 2), (CF1, 2), (DENT, 2)],
)
def test_fixed_objective_count_is_warned_and_ignored(problem_type: type[MOP], expected_n_obj: int) -> None:
    with pytest.warns(UserWarning, match=rf"fixed n_obj={expected_n_obj}"):
        problem = problem_type(n_obj=99)

    assert problem.n_obj == expected_n_obj


def test_explicit_default_objective_count_still_warns() -> None:
    with pytest.warns(UserWarning, match=r"fixed n_obj=2"):
        problem = UF1(n_obj=2)

    assert problem.n_obj == 2


@pytest.mark.parametrize("problem_type", [UF1, UF8, ZDT1, CF1, DTLZ1, DENT, CONV3])
def test_problem_constructor_starts_with_base_argument_order(problem_type: type[MOP]) -> None:
    parameters = list(inspect.signature(problem_type).parameters)
    assert parameters[:4] == ["n_var", "n_obj", "xl", "xu"]


@pytest.mark.parametrize("problem_type", [UF1, ZDT1, CF1, DTLZ1])
def test_scalar_bounds_are_broadcast_by_mop(problem_type: type[MOP]) -> None:
    problem = problem_type(n_var=5, xl=-2.5, xu=3.5)

    np.testing.assert_array_equal(problem.xl, np.full(5, -2.5))
    np.testing.assert_array_equal(problem.xu, np.full(5, 3.5))


def test_effective_constraint_counts_are_instance_local() -> None:
    plain = CF1()
    with_bounds = CF1(boundary_constraints=True)

    assert CF1.default_n_ieq_constr == 1
    assert plain.n_ieq_constr == 1
    assert with_bounds.n_ieq_constr == 1 + 2 * with_bounds.n_var


def test_resolved_metadata_are_instance_annotations() -> None:
    assert {"n_obj", "n_var", "xl", "xu"}.isdisjoint(MOP.__annotations__)
    problem = ZDT1(n_var=5)
    assert {"n_obj", "n_var", "xl", "xu"} <= problem.__dict__.keys()


def test_boundary_constraints_is_only_a_boolean_box_constraint_switch() -> None:
    without_bounds = UF1()
    with_bounds = UF1(boundary_constraints=True)

    assert without_bounds.n_ieq_constr == 0
    assert with_bounds.n_ieq_constr == 2 * with_bounds.n_var
    for invalid_value in (None, 0, 1, "true"):
        with pytest.raises(TypeError, match="must be a Boolean"):
            UF1(boundary_constraints=invalid_value)


@pytest.mark.parametrize(("name", "value"), [("n_var", 0), ("n_var", True), ("n_obj", -1)])
def test_mop_rejects_invalid_dimensions(name: str, value: object) -> None:
    arguments = {"n_var": 2, "n_obj": 2, "xl": 0.0, "xu": 1.0, name: value}

    with pytest.raises(ValueError, match=rf"`{name}` must be a positive integer"):
        _ToyMOP(**arguments)


def test_mop_rejects_incompatible_or_reversed_bounds() -> None:
    with pytest.raises(ValueError, match="scalar or have shape"):
        _ToyMOP(n_var=3, xl=[0.0, 0.0], xu=1.0)
    with pytest.raises(ValueError, match="lower bound"):
        _ToyMOP(xl=2.0, xu=1.0)


def test_boundary_constraints_append_box_constraints_after_native_constraints() -> None:
    problem = _ToyCMOP(boundary_constraints=True)
    x = np.array([0.25, 0.75])

    expected = np.r_[np.sum(x) - 0.5, problem.xl - x, x - problem.xu]
    np.testing.assert_allclose(problem.ieq_constraint(x), expected)
    assert problem.n_eq_constr == 1
    assert problem.n_ieq_constr == 1 + 2 * problem.n_var


def test_cmop_rejects_negative_constraint_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CMOP(n_var=2, n_obj=2, xl=0.0, xu=1.0, n_ieq_constr=-1)
