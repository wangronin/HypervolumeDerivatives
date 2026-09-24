import numpy as np
import pytest

from hvd.base import State
from hvd.problems import CF1, Eq1DTLZ1, ZDT1


class _ProblemFunctions:
    def __init__(self) -> None:
        self.single_calls = 0
        self.batch_calls: dict[str, int] = {}

    def _record_call(self, name: str, x: np.ndarray) -> None:
        if x.ndim == 1:
            self.single_calls += 1
        else:
            self.batch_calls[name] = self.batch_calls.get(name, 0) + 1

    def objective(self, x: np.ndarray) -> np.ndarray:
        self._record_call("objective", x)
        return np.stack((np.sum(x, axis=-1), np.sum(x**2, axis=-1)), axis=-1)

    def objective_jacobian(self, x: np.ndarray) -> np.ndarray:
        self._record_call("objective_jacobian", x)
        return np.stack((np.ones_like(x), 2 * x), axis=-2)

    def eq_constraint(self, x: np.ndarray) -> np.ndarray:
        self._record_call("eq_constraint", x)
        return np.sum(x, axis=-1, keepdims=True)

    def eq_jacobian(self, x: np.ndarray) -> np.ndarray:
        self._record_call("eq_jacobian", x)
        return np.ones((*x.shape[:-1], 1, x.shape[-1]))

    def eq_hessian(self, x: np.ndarray) -> np.ndarray:
        self._record_call("eq_hessian", x)
        return np.zeros((*x.shape[:-1], 1, x.shape[-1], x.shape[-1]))

    def ieq_constraint(self, x: np.ndarray) -> np.ndarray:
        self._record_call("ieq_constraint", x)
        return x[..., :1] - 1

    def ieq_jacobian(self, x: np.ndarray) -> np.ndarray:
        self._record_call("ieq_jacobian", x)
        return np.broadcast_to([[1.0, 0.0]], (*x.shape[:-1], 1, x.shape[-1]))

    def ieq_hessian(self, x: np.ndarray) -> np.ndarray:
        self._record_call("ieq_hessian", x)
        return np.zeros((*x.shape[:-1], 1, x.shape[-1], x.shape[-1]))


def test_state_passes_populations_and_points_directly_to_callbacks() -> None:
    functions = _ProblemFunctions()
    state = State(
        n_var=2,
        n_eq=1,
        n_ieq=1,
        func=functions.objective,
        jac=functions.objective_jacobian,
        h=functions.eq_constraint,
        h_jac=functions.eq_jacobian,
        h_hess=functions.eq_hessian,
        g=functions.ieq_constraint,
        g_jac=functions.ieq_jacobian,
        g_hess=functions.ieq_hessian,
    )
    x = np.array([[0.25, 0.5], [1.5, -0.5]])

    state.update(x)

    np.testing.assert_allclose(state.Y, [[0.75, 0.3125], [1.0, 2.5]])
    assert state.J.shape == (2, 2, 2)
    np.testing.assert_allclose(state.cstr_value, [[0.75, -0.75], [1.0, 0.5]])
    np.testing.assert_array_equal(state.active_indices, [[True, False], [True, True]])
    assert state.cstr_grad.shape == (2, 2, 2)
    assert state.cstr_hess.shape == (2, 2, 2, 2)
    assert functions.single_calls == 0
    assert functions.batch_calls == {
        "objective": 1,
        "objective_jacobian": 1,
        "eq_constraint": 1,
        "eq_jacobian": 1,
        "eq_hessian": 1,
        "ieq_constraint": 1,
        "ieq_jacobian": 1,
        "ieq_hessian": 1,
    }
    assert state.n_jac_evals == len(x)
    assert state.n_cstr_jac_evals == len(x)
    assert state.n_cstr_hess_evals == len(x)

    batch_calls = functions.batch_calls.copy()
    state.update_one(np.array([0.1, 0.2]), 0)
    assert functions.single_calls == 8
    assert functions.batch_calls == batch_calls
    assert state.n_jac_evals == len(x) + 1


@pytest.mark.parametrize(
    "problem_type,boundary_constraints", [(ZDT1, False), (ZDT1, True), (CF1, False), (Eq1DTLZ1, False)]
)
def test_state_handles_callable_methods_for_absent_constraint_families(problem_type, boundary_constraints):
    problem = problem_type(n_var=5, boundary_constraints=boundary_constraints)
    state = State(
        problem.n_var,
        problem.n_eq_constr,
        problem.n_ieq_constr,
        problem.objective,
        problem.objective_jacobian,
        h=problem.eq_constraint,
        h_jac=problem.eq_jacobian,
        h_hess=problem.eq_hessian,
        g=problem.ieq_constraint,
        g_jac=problem.ieq_jacobian,
        g_hess=problem.ieq_hessian,
    )
    population = np.full((2, problem.n_var), 0.3)
    state.update(population)
    state.update_one(population[0], 0)

    n_cstr = problem.n_eq_constr + problem.n_ieq_constr
    assert state.cstr_value.shape == state.active_indices.shape == (2, n_cstr)
    assert state.cstr_grad.shape == (2, n_cstr, problem.n_var)
    assert state.cstr_hess.shape == (2, n_cstr, problem.n_var, problem.n_var)
    assert state._constrained == (n_cstr > 0)
    assert state.n_cstr_jac_evals == state.n_cstr_hess_evals == (3 if n_cstr else 0)
    assert np.all(np.isfinite(state.cstr_grad))
    assert np.all(np.isfinite(state.cstr_hess))


def test_check_kkt_for_unconstrained_multiobjective_points() -> None:
    def objective(x: np.ndarray) -> np.ndarray:
        return np.stack((x[..., 0] ** 2, (x[..., 0] - 2) ** 2), axis=-1)

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.stack((2 * x[..., 0], 2 * (x[..., 0] - 2)), axis=-1)[..., None]

    state = State(1, 0, 0, objective, jacobian)
    state.update(np.array([[1.0], [-1.0]]))

    np.testing.assert_array_equal(state.check_KKT(), [True, False])


def test_check_kkt_allows_free_equality_multipliers_and_requires_feasibility() -> None:
    def objective(x: np.ndarray) -> np.ndarray:
        return x[..., :1]

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.ones((*x.shape[:-1], 1, 1))

    def equality(x: np.ndarray) -> np.ndarray:
        return x[..., :1] - 1

    def equality_jacobian(x: np.ndarray) -> np.ndarray:
        return np.ones((*x.shape[:-1], 1, 1))

    state = State(1, 1, 0, objective, jacobian, h=equality, h_jac=equality_jacobian)
    state.update(np.array([[1.0], [0.0]]))

    np.testing.assert_array_equal(state.check_KKT(), [True, False])


def test_check_kkt_enforces_nonnegative_active_inequality_multipliers() -> None:
    def objective(x: np.ndarray) -> np.ndarray:
        return x[..., :1]

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.ones((*x.shape[:-1], 1, 1))

    def inequality(x: np.ndarray) -> np.ndarray:
        return -x[..., :1]

    def inequality_jacobian(x: np.ndarray) -> np.ndarray:
        return -np.ones((*x.shape[:-1], 1, 1))

    state = State(1, 0, 1, objective, jacobian, g=inequality, g_jac=inequality_jacobian)
    state.update(np.array([[0.0], [1.0], [-1.0]]))

    np.testing.assert_array_equal(state.check_KKT(), [True, False, False])
