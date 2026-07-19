import numpy as np
import pytest

from hvd.problems import (
    MOP,
    CMOP,
    WFG1,
    WFG2,
    WFG3,
    WFG4,
    WFG5,
    WFG6,
    WFG7,
    WFG8,
    WFG9,
    ZDT1,
    ZDT2,
    ZDT3,
    ZDT4,
    ZDT6,
    get_pymoo_problem,
    pymoo_problem_names,
)


def test_native_problem_as_pymoo_problem():
    problem = ZDT1(n_var=3)
    pymoo_problem = problem.as_pymoo_problem()
    X = np.array([[0.2, 0.1, 0.2], [0.4, 0.2, 0.1]])
    np.testing.assert_allclose(pymoo_problem.evaluate(X), problem.objective_batch(X))


def test_from_pymoo_is_idempotent_for_native_problem():
    problem = ZDT1()
    assert MOP.from_pymoo(problem) is problem


def test_selected_pymoo_problem_factory_is_differentiable_and_restores_backend():
    import pymoo.gradient as gradient

    previous_backend = gradient.TOOLBOX
    problem = get_pymoo_problem("ZDT1", n_var=5)
    x = np.linspace(0.1, 0.5, problem.n_var)

    assert gradient.TOOLBOX == previous_backend
    assert problem.n_ieq_constr == 2 * problem.n_var
    assert problem.objective_jacobian(x).shape == (problem.n_obj, problem.n_var)
    assert problem.ieq_jacobian(x).shape == (problem.n_ieq_constr, problem.n_var)
    np.testing.assert_allclose(
        problem.as_pymoo_problem().evaluate(x, return_values_of=["F"]),
        problem.objective(x),
    )


def test_pymoo_problem_factory_rejects_unselected_problem():
    assert "zdt1" in pymoo_problem_names()
    assert "wfg1" not in pymoo_problem_names()
    with pytest.raises(ValueError, match="Unsupported pymoo problem"):
        get_pymoo_problem("not-selected")


@pytest.mark.parametrize("problem_type", [ZDT1, ZDT2, ZDT3, ZDT4, ZDT6])
def test_native_zdt_matches_factory_values_and_derivatives(problem_type: type[MOP]):
    native = problem_type(n_var=10)
    wrapped = get_pymoo_problem(problem_type.__name__, n_var=10)
    x = np.linspace(0.15, 0.85, native.n_var)

    np.testing.assert_allclose(native.objective(x), wrapped.objective(x))
    np.testing.assert_array_equal(native.xl, wrapped.xl)
    np.testing.assert_array_equal(native.xu, wrapped.xu)
    np.testing.assert_allclose(native.get_pareto_front(), wrapped.get_pareto_front())
    np.testing.assert_allclose(native.objective_jacobian(x), wrapped.objective_jacobian(x))
    np.testing.assert_allclose(
        native.objective_hessian(x),
        wrapped.objective_hessian(x),
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "problem_type",
    [WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9],
)
def test_native_wfg_matches_pymoo_and_supports_ad(problem_type: type[CMOP]):
    from pymoo.problems import get_problem

    problem = problem_type(n_var=6, n_obj=3)
    raw = get_problem(problem_type.__name__, n_var=6, n_obj=3)
    x = np.linspace(0.17, 0.73, raw.n_var) * raw.xu

    np.testing.assert_allclose(
        problem.objective(x),
        raw.evaluate(x, return_values_of=["F"]),
        rtol=2e-6,
        atol=2e-6,
    )
    assert isinstance(problem, CMOP)
    assert problem.n_ieq_constr == 0
    assert np.all(np.isfinite(problem.objective_jacobian(x)))
    assert np.all(np.isfinite(problem.objective_hessian(x)))


def test_native_wfg_can_add_box_constraints() -> None:
    problem = WFG1(n_var=6, n_obj=3, boundary_constraints=True)
    assert problem.n_ieq_constr == 2 * problem.n_var


def test_jax_compatible_pymoo_problem_as_native_mop():
    import jax.numpy as jnp
    from pymoo.core.problem import Problem

    class QuadraticProblem(Problem):
        def __init__(self):
            super().__init__(n_var=2, n_obj=2, xl=-1, xu=1)

        def _evaluate(self, X: jnp.ndarray, out: dict[str, jnp.ndarray], *args, **kwargs) -> None:
            out["F"] = jnp.column_stack((jnp.sum(X**2, axis=1), jnp.sum((X - 1) ** 2, axis=1)))

    problem = MOP.from_pymoo(QuadraticProblem(), boundary_constraints=False)
    x = np.array([0.25, -0.5])
    assert problem.objective(x).shape == (2,)
    assert problem.objective_jacobian(x).shape == (2, 2)
    assert problem.objective_hessian(x).shape == (2, 2, 2)


def test_non_jax_pymoo_problem_fails_during_adaptation():
    from pymoo.core.problem import Problem

    class NumPyProblem(Problem):
        def __init__(self):
            super().__init__(n_var=2, n_obj=1, xl=-1, xu=1)

        def _evaluate(self, X: np.ndarray, out: dict[str, np.ndarray], *args, **kwargs) -> None:
            values = np.asarray(X)
            out["F"] = np.sum(values**2, axis=1, keepdims=True)

    with pytest.raises(TypeError, match="not JAX-traceable"):
        MOP.from_pymoo(NumPyProblem())


def test_pymoo_adapter_rejects_non_mop_problem():
    from hvd.problems._pymoo_wrapper import _PymooProblemAdapter

    with pytest.raises(TypeError, match="Expected an MOP"):
        _PymooProblemAdapter(object())
