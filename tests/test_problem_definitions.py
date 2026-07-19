import inspect

import numpy as np
import pytest

from hvd.problems import CF1, DTLZ1, Eq1DTLZ1, MOP, UF1, UF8, ZDT1
from hvd.problems.misc import CONV3, DENT


@pytest.mark.parametrize(
    ("problem_type", "expected_n_obj"),
    [(UF1, 2), (UF8, 3), (ZDT1, 2), (CF1, 2), (DENT, 2)],
)
def test_fixed_objective_count_is_warned_and_ignored(
    problem_type: type[MOP], expected_n_obj: int
) -> None:
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
    with pytest.raises(TypeError, match="must be a Boolean"):
        UF1(boundary_constraints=None)


def test_batched_objective_derivatives_match_single_evaluations() -> None:
    problem = ZDT1(n_var=5)
    x = np.linspace(0.1, 0.9, problem.n_var)
    X = np.stack((x, 0.9 * x))

    expected_jacobians = np.stack([problem.objective_jacobian(row) for row in X])
    expected_hessians = np.stack([problem.objective_hessian(row) for row in X])
    np.testing.assert_allclose(problem.objective_jacobian_batch(X), expected_jacobians)
    np.testing.assert_allclose(problem.objective_hessian_batch(X), expected_hessians)


def test_batched_constraint_derivatives_match_single_evaluations() -> None:
    ieq_problem = CF1(n_var=5)
    eq_problem = Eq1DTLZ1(n_var=5)
    X = np.stack((np.full(5, 0.3), np.full(5, 0.6)))

    np.testing.assert_allclose(
        ieq_problem.ieq_jacobian_batch(X),
        np.stack([ieq_problem.ieq_jacobian(row) for row in X]),
    )
    np.testing.assert_allclose(
        ieq_problem.ieq_hessian_batch(X),
        np.stack([ieq_problem.ieq_hessian(row) for row in X]),
    )
    np.testing.assert_allclose(
        eq_problem.eq_jacobian_batch(X),
        np.stack([eq_problem.eq_jacobian(row) for row in X]),
    )
    np.testing.assert_allclose(
        eq_problem.eq_hessian_batch(X),
        np.stack([eq_problem.eq_hessian(row) for row in X]),
    )
