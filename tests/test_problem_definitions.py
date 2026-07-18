import inspect

import numpy as np
import pytest

from hvd.problems import CF1, DTLZ1, MOP, UF1, UF8, ZDT1
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
