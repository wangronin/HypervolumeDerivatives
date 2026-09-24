import inspect

import numpy as np
import pytest

import hvd.newton as newton_module
from hvd.mmd import MMD
from hvd.mmd_newton import MMDN
from hvd.newton import HVN, DpN
from hvd.problems import ZDT1
from hvd.reference_set import ReferenceSet


def test_box_projection_is_opt_in_for_mmdn_and_dpn() -> None:
    assert inspect.signature(MMDN).parameters["project_box_constraints"].default is False
    assert inspect.signature(DpN).parameters["project_box_constraints"].default is False


@pytest.mark.parametrize("optimizer_type", [HVN, DpN, MMDN])
@pytest.mark.parametrize("boundary_constraints", [False, True])
def test_newton_detects_constraint_counts_from_optional_method_results(optimizer_type, boundary_constraints):
    problem = ZDT1(n_var=3, boundary_constraints=boundary_constraints)
    x0 = np.full((2, problem.n_var), 0.3)
    options = dict(
        n_obj=problem.n_obj,
        func=problem.objective,
        jac=problem.objective_jacobian,
        xl=problem.xl,
        xu=problem.xu,
        verbose=False,
        h=problem.eq_constraint,
        h_jac=problem.eq_jacobian,
        h_hessian=problem.eq_hessian,
        g=problem.ieq_constraint,
        g_jac=problem.ieq_jacobian,
        g_hessian=problem.ieq_hessian,
    )
    if optimizer_type is MMDN:
        options.update(
            n_var=problem.n_var,
            X0=x0,
            indicator=MMD(
                problem.n_var, problem.n_obj, problem.get_pareto_front(3),
                problem.objective, problem.objective_jacobian, problem.objective_hessian,
            ),
        )
    elif optimizer_type is HVN:
        options.update(
            n_var=problem.n_var, X0=x0, ref=np.array([2.0, 10.0]), hessian=problem.objective_hessian
        )
    else:
        options.update(
            dim=problem.n_var,
            x0=x0,
            ref=ReferenceSet(problem.get_pareto_front(3)),
            hessian=problem.objective_hessian,
        )
    optimizer = optimizer_type(**options)

    assert optimizer.n_eq == 0
    assert optimizer.n_ieq == problem.n_ieq_constr
    assert optimizer._constrained == boundary_constraints
    assert optimizer.state.cstr_grad.shape == (len(x0), problem.n_ieq_constr, problem.n_var)


def test_dpn_uses_blockwise_hessian_regularization(monkeypatch) -> None:
    problem = ZDT1(n_var=3)
    x0 = problem.get_pareto_set(3)
    x0[:, 1:] += 0.02
    calls = []

    def regularize(block: np.ndarray, block_size: int) -> np.ndarray:
        calls.append((block.shape, block_size))
        return np.eye(block_size)

    monkeypatch.setattr(newton_module, "regularize_hessian_block", regularize)
    optimizer = DpN(
        dim=problem.n_var,
        n_obj=problem.n_obj,
        ref=ReferenceSet(problem.get_pareto_front(3)),
        func=problem.objective,
        jac=problem.objective_jacobian,
        hessian=problem.objective_hessian,
        x0=x0,
        xl=problem.xl,
        xu=problem.xu,
        regularization=True,
        verbose=False,
    )

    optimizer._compute_indicator_value(optimizer.state.Y)
    optimizer._compute_netwon_step(optimizer.state)

    assert calls == [((problem.n_var, problem.n_var), problem.n_var)] * len(x0)
