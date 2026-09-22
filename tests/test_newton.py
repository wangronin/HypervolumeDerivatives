import inspect

import numpy as np

import hvd.newton as newton_module
from hvd.mmd_newton import MMDNewton
from hvd.newton import DpN
from hvd.problems import ZDT1
from hvd.reference_set import ReferenceSet


def test_box_projection_is_opt_in_for_mmdn_and_dpn() -> None:
    assert inspect.signature(MMDNewton).parameters["project_box_constraints"].default is False
    assert inspect.signature(DpN).parameters["project_box_constraints"].default is False


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
