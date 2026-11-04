"""Run one MMD Newton configuration on stored MOEA populations and save one CSV."""

import argparse
import json
import random
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
from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.hypervolume import HV
from hvd.mmd import MMD, MMDMatching
from hvd.mmd.kernels import RBF, RationalQuadratic
from hvd.mmd_newton import MMDN
from hvd.reference_set import ReferenceSet
from hvd.utils import get_non_dominated
from scripts.utils import (
    get_data_prefix,
    get_pareto_front,
    get_run_instances,
    kernel_theta,
    read_reference_set_data,
)

PROBLEM_NAMES = [
    name
    for name in problems.__all__
    if isinstance(getattr(problems, name), type) and name not in {"MOP", "CMOP"}
]
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return parser


def execute(run, args, configuration, problem, pareto_front, ref_point):
    random.seed(66 + run * 1009)
    np.random.seed(66 + run * 1009)
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
    optimizer = MMDN(
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
    _, y, _ = optimizer.run()
    if not np.isfinite(y).all():
        raise ValueError(f"Run {run}: nonfinite final objectives")
    y = get_non_dominated(y)
    performance = {
        "GD": GenerationalDistance(pareto_front).compute(Y=y),
        "IGD": InvertedGenerationalDistance(pareto_front).compute(Y=y),
        "HV": HV(ref_point).compute(Y=y),
    }
    return {"run": run, **performance, "Jac_calls": optimizer.state.n_jac_evals}


def get_config(args):
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


def main():
    args = build_parser().parse_args()
    configuration, args = get_config(args)
    instances = get_run_instances(args.problem, args.algorithm, args.generation, args.data_path)
    if not instances:
        raise FileNotFoundError(
            f"No {args.problem} / {args.algorithm} runs at generation {args.generation} in {args.data_path}."
        )
    prefix = get_data_prefix(args.problem, args.algorithm, instances[0], args.generation, args.data_path)
    pop_x_file = args.data_path / f"{prefix}_lastpopu_x_gen{args.generation}.csv"
    pop_y_file = args.data_path / f"{prefix}_lastpopu_y_gen{args.generation}.csv"
    n_var = pd.read_csv(pop_x_file, header=None, nrows=1).shape[1]
    n_obj = pd.read_csv(pop_y_file, header=None, nrows=1).shape[1]
    problem = getattr(problems, args.problem)(
        n_var=n_var, n_obj=n_obj, boundary_constraints=BOUNDARY_CONSTRAINTS
    )
    pareto_front = get_pareto_front(problem)
    ref_point = pd.read_csv(ROOT / "scripts" / "ref_point.csv", index_col="problem")
    ref_point = ref_point.loc[args.problem].dropna().to_numpy(dtype=float)
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
