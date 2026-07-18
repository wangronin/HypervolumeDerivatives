import pytest

from hvd.problems import CF1, MOP, UF1, UF8, ZDT1
from hvd.problems.misc import DENT


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
