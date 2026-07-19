import numpy as np

from hvd.base import State


class _BatchedFunctions:
    def __init__(self) -> None:
        self.single_calls = 0
        self.batch_calls: dict[str, int] = {}

    def _record_batch(self, name: str) -> None:
        self.batch_calls[name] = self.batch_calls.get(name, 0) + 1

    def objective(self, x: np.ndarray) -> np.ndarray:
        self.single_calls += 1
        return np.array([np.sum(x), np.sum(x**2)])

    def objective_batch(self, x: np.ndarray) -> np.ndarray:
        self._record_batch("objective")
        return np.column_stack((np.sum(x, axis=1), np.sum(x**2, axis=1)))

    def objective_jacobian(self, x: np.ndarray) -> np.ndarray:
        self.single_calls += 1
        return np.stack((np.ones(2), 2 * x))

    def objective_jacobian_batch(self, x: np.ndarray) -> np.ndarray:
        self._record_batch("objective_jacobian")
        return np.stack((np.ones_like(x), 2 * x), axis=1)

    def eq_constraint(self, x: np.ndarray) -> np.ndarray:
        self.single_calls += 1
        return np.array([np.sum(x)])

    def eq_constraint_batch(self, x: np.ndarray) -> np.ndarray:
        self._record_batch("eq_constraint")
        return np.sum(x, axis=1, keepdims=True)

    def eq_jacobian(self, x: np.ndarray) -> np.ndarray:
        self.single_calls += 1
        return np.ones((1, len(x)))

    def eq_jacobian_batch(self, x: np.ndarray) -> np.ndarray:
        self._record_batch("eq_jacobian")
        return np.ones((len(x), 1, x.shape[1]))

    def eq_hessian(self, x: np.ndarray) -> np.ndarray:
        self.single_calls += 1
        return np.zeros((1, len(x), len(x)))

    def eq_hessian_batch(self, x: np.ndarray) -> np.ndarray:
        self._record_batch("eq_hessian")
        return np.zeros((len(x), 1, x.shape[1], x.shape[1]))

    def ieq_constraint(self, x: np.ndarray) -> np.ndarray:
        self.single_calls += 1
        return np.array([x[0] - 1])

    def ieq_constraint_batch(self, x: np.ndarray) -> np.ndarray:
        self._record_batch("ieq_constraint")
        return x[:, :1] - 1

    def ieq_jacobian(self, x: np.ndarray) -> np.ndarray:
        self.single_calls += 1
        return np.array([[1.0, 0.0]])

    def ieq_jacobian_batch(self, x: np.ndarray) -> np.ndarray:
        self._record_batch("ieq_jacobian")
        result = np.zeros((len(x), 1, x.shape[1]))
        result[:, 0, 0] = 1
        return result

    def ieq_hessian(self, x: np.ndarray) -> np.ndarray:
        self.single_calls += 1
        return np.zeros((1, len(x), len(x)))

    def ieq_hessian_batch(self, x: np.ndarray) -> np.ndarray:
        self._record_batch("ieq_hessian")
        return np.zeros((len(x), 1, x.shape[1], x.shape[1]))


def test_state_update_uses_available_batch_variants() -> None:
    functions = _BatchedFunctions()
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


def test_check_kkt_for_unconstrained_multiobjective_points() -> None:
    def objective(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2, (x[0] - 2) ** 2])

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.array([[2 * x[0]], [2 * (x[0] - 2)]])

    state = State(1, 0, 0, objective, jacobian)
    state.update(np.array([[1.0], [-1.0]]))

    np.testing.assert_array_equal(state.check_KKT(), [True, False])


def test_check_kkt_allows_free_equality_multipliers_and_requires_feasibility() -> None:
    def objective(x: np.ndarray) -> np.ndarray:
        return np.array([x[0]])

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.array([[1.0]])

    def equality(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] - 1])

    def equality_jacobian(x: np.ndarray) -> np.ndarray:
        return np.array([[1.0]])

    state = State(1, 1, 0, objective, jacobian, h=equality, h_jac=equality_jacobian)
    state.update(np.array([[1.0], [0.0]]))

    np.testing.assert_array_equal(state.check_KKT(), [True, False])


def test_check_kkt_enforces_nonnegative_active_inequality_multipliers() -> None:
    def objective(x: np.ndarray) -> np.ndarray:
        return np.array([x[0]])

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.array([[1.0]])

    def inequality(x: np.ndarray) -> np.ndarray:
        return np.array([-x[0]])

    def inequality_jacobian(x: np.ndarray) -> np.ndarray:
        return np.array([[-1.0]])

    state = State(1, 0, 1, objective, jacobian, g=inequality, g_jac=inequality_jacobian)
    state.update(np.array([[0.0], [1.0], [-1.0]]))

    np.testing.assert_array_equal(state.check_KKT(), [True, False, False])
