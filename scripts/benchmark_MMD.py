"""Run one MMD Newton configuration on stored MOEA populations and save one CSV."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from jax import config as jax_config
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
jax_config.update("jax_enable_x64", True)

from hvd import problems
from hvd.mmd import MMD, MMDMatching
from hvd.mmd.kernels import RBF, RationalQuadratic
from hvd.mmd_newton import MMDN
from hvd.reference_set import ReferenceSet
from scripts.utils import (
    MMDConfig,
    PROBLEM_NAMES,
    compute_performance,
    get_dimensions_from_data,
    get_pareto_front,
    get_reference_point,
    get_run_instances,
    kernel_theta,
    read_reference_set_data,
    set_random_seed,
)

BOUNDARY_CONSTRAINTS = True
DEFAULT_NEWTON_MAX_ITERS = 5
DEFAULT_MOEA_GENERATION = 600


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=PROBLEM_NAMES)
    parser.add_argument("--algorithm", choices=["NSGA-II", "NSGA-III", "MOEAD"], default="NSGA-III")
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--generation", type=int, default=DEFAULT_MOEA_GENERATION)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_NEWTON_MAX_ITERS)
    parser.add_argument("--regularization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--project-box-constraints", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--matching", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--ref-point", nargs="+", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return parser


def get_optimizer(
    run: int, args: argparse.Namespace, configuration: MMDConfig, problem: problems.CMOP
) -> MMDN:
    """Build the configured MMD optimizer from one stored run."""
    set_random_seed(66 + run * 1009)
    ref, x0, y0, labels, eta = read_reference_set_data(
        args.data_path, args.problem, args.algorithm, run, args.generation, args.matching
    )
    reference_set = ReferenceSet(ref=ref, eta=eta, Y_idx=labels)
    theta = kernel_theta(configuration["theta_multiplier"], y0, np.vstack(list(ref.values())))
    kernel = {"rbf": RBF, "rational_quadratic": RationalQuadratic}[configuration["kernel"]](theta=theta)
    indicator = (MMDMatching if args.matching else MMD)(
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        ref=reference_set,
        func=problem.objective,
        jac=problem.objective_jacobian,
        hessian=problem.objective_hessian,
        kernel=kernel,
        **({"beta": configuration.get("beta", 0.25)} if args.matching else {}),
    )
    return MMDN(
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        indicator=indicator,
        func=problem.objective,
        jac=problem.objective_jacobian,
        h=problem.eq_constraint,
        h_jac=problem.eq_jacobian,
        h_hessian=problem.eq_hessian,
        g=problem.ieq_constraint,
        g_jac=problem.ieq_jacobian,
        g_hessian=problem.ieq_hessian,
        N=len(x0),
        X0=x0,
        xl=problem.xl,
        xu=problem.xu,
        max_iters=args.max_iters,
        regularization=args.regularization,
        project_box_constraints=args.project_box_constraints,
        verbose=False,
    )


def execute(
    run: int,
    args: argparse.Namespace,
    configuration: MMDConfig,
    problem: problems.CMOP,
    pareto_front: np.ndarray,
    ref_point: np.ndarray,
) -> dict[str, float | int]:
    optimizer = get_optimizer(run, args, configuration, problem)
    _, y, _ = optimizer.run()
    performance = compute_performance(y, pareto_front, ref_point)
    return {"run": run, **performance, "Jac_calls": optimizer.state.n_jac_evals}


def get_config(args: argparse.Namespace) -> tuple[MMDConfig, argparse.Namespace]:
    config_file = json.loads(args.config_file.read_text())
    configuration = config_file.get("best_config", config_file)
    if args.regularization is None:
        args.regularization = configuration["regularization"]
    if args.project_box_constraints is None:
        args.project_box_constraints = config_file.get("project_box_constraints", True)
    if config_file.get("indicator") == "MMDMatching":
        args.matching = True
    for key, expected in (("problem", args.problem), ("algorithm", args.algorithm)):
        if key in config_file and config_file[key] != expected:
            raise ValueError(f"Configuration {key} does not match {expected}")
    return configuration, args


def main() -> None:
    args = build_parser().parse_args()
    configuration, args = get_config(args)
    instances = get_run_instances(args.problem, args.algorithm, args.generation, args.data_path)
    if not instances:
        raise FileNotFoundError(
            f"No {args.problem} / {args.algorithm} runs at generation {args.generation} in {args.data_path}."
        )
    n_var, n_obj = get_dimensions_from_data(args, instances[0])
    problem = getattr(problems, args.problem)(
        n_var=n_var, n_obj=n_obj, boundary_constraints=BOUNDARY_CONSTRAINTS
    )
    pareto_front = get_pareto_front(problem)
    ref_point = get_reference_point(args, problem)
    # create folders
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"optimize {args.problem}")
    print(f"JAX 64-bit enabled: {jax_config.jax_enable_x64}")
    print(f"parallel benchmark workers: {args.n_jobs}")
    params = (args, configuration, problem, pareto_front, ref_point)
    if args.n_jobs == 1:
        data = [execute(run, *params) for run in instances]
    else:
        data = Parallel(n_jobs=args.n_jobs)(delayed(execute)(run, *params) for run in instances)

    columns = ["run", "HV", "IGD", "GD", "Jac_calls"]
    alg_name = "MMDNMatching" if args.matching else "MMDN"
    output = args.output or args.results_dir / f"{args.problem}-{alg_name}-{args.algorithm}-{args.generation}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, columns=columns).to_csv(output, index=False)


if __name__ == "__main__":
    main()
