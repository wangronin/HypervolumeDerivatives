# Multi-objective Newton Methods

This package contains three set-oriented Newton methods for solving _constrainted continuous multiobjective optimization problems_:

* **What for?**
  * a **MMD-based Newton Method** (MMDN), which is submitted to _NeurIPS 2025 conference_.
  * a **$\Delta_p$ Newton Method** (DpN), which minimizes average Hausdorff distance, accepted in IEEE TEVC [WRU+24].
  * a **Hypervolume Newton Method** (HVN), which maximizes the Hypervolume indicator, accepted in IEEE TCYB and other venues [DEW22,SSW+22].
* **Why?**
  * the Newton method has a local **quadratic convergence** under some mild condition of objective functions.
  * Perhaps, you'd like to refine the final outcome of some direct optimizers with the Hypervolume Newton method..
* **When to use it?** the objective function is at least _twice continuously differentiable_.

Specifically, you will find the following major functionalities:

1. `hvd.mmd_newton.MMDNewton`: the MMD-based Newton method, submitted to NeurIPS 2025.

## References

Will show up after double-blind reviewing.

## Installation

For now, please take the lastest version from the `neurips2025` branch:

```shell
cd HypervolumeDerivatives && pip install -f requirements.txt
```

You can decide to use a Python virtual environment to install the dependencies.

## Reproducing NeurIPS 2025's results

The experimental data needed for running MMD-based Newton (MMDN) can be accessed via [`MMD_data.zip`](https://drive.google.com/file/d/1OwDs89y1ccGbBSxG-4qpbLxtgE7e9x5U/view?usp=sharing). Please download and unzip it before reproducing the experiments. After unzipping, you should have a `./MMD_data` folder containing CSV files of initial points and reference sets generated from multi-objective optimization evolutionary algorithms (MOEAs). See our experimental procedure in the submitted paper for details.

In the `/scripts` folder, you can find the experimental scripts for our NeurIPS 2025 submission: `/scripts/benchmark_MMD.py` performs the experiments in Sec. 6 of the paper. Calling signature is:

```shell
python ./scripts/benchmark_MMD.py ZDT1
```

`ZDT1` is one of the test problem we had in the experiments among `ZDT1, ZDT2, ZDT3, ZDT4, DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ1, DTLZ6, DTLZ7`. This script will run in parallel by default and save a CSV and LaTeX table for the results in the `./result` folder.

The baseline MOEAs should be executed with the folllowing command:

```shell
python ./scripts/benchmark_EA.py ZDT1
```

This script will run in parallel by default and save a CSV and LaTeX table for the results in the `./result` folder. After benchmarking both MMDN and MOEAs on each test problems. The statistical summary/hypothesis testing is performed by running

```shell
python ./scripts/compute_statistics_MMD.py
```

which take the raw data stored in `./.results` and perform Mann-Whitney U test with multiple testing corrections. The test results are save in files named `MMD-300.txt` and `MMD-300.tex`.


### Example Usage of MMDN

```Python
from hvd.delta_p import GenerationalDistance, InvertedGenerationalDistance
from hvd.mmd_newton import MMDNewton
from hvd.problems import ZDT1, PymooProblemWithAD
from hvd.reference_set import ReferenceSet

max_iters = 15
# create ZDT1 problem; search dimension = 10
problem = PymooProblemWithAD(ZDT1(n_var=10))
# get Pareto front
pareto_front = problem.get_pareto_front(1000)
# reference set is initially on the Pareto front
# it will be shifted by `0.08 * eta` in the first iteration
reference_set = problem.get_pareto_front(15)
# initial approximation set close to the efficient set
X0 = problem.get_pareto_set(15, kind="linear")
X0[:, 1] += 0.02
Y0 = np.array([problem.objective(x) for x in X0])
# shifting direction of the reference set
eta = {0: np.array([-0.70710678, -0.70710678])}
# performance metrics
metrics = dict(
    GD=GenerationalDistance(pareto_front),
    IGD=InvertedGenerationalDistance(pareto_front)
)
opt = MMDNewton(
    n_var=problem.n_var, # search dimension
    n_obj=problem.n_obj, # number of objectives
    ref=ReferenceSet(ref=reference_set, eta=eta),
    func=problem.objective, # objective function 
    jac=problem.objective_jacobian, # objective Jacobian 
    hessian=problem.objective_hessian, # objective Hessian
    g=problem.ieq_constraint, # inequality constraint
    g_jac=problem.ieq_jacobian, # inequality Jacobian
    N=len(X0),
    X0=X0, # initial points 
    xl=problem.xl, # search space: lower bounds
    xu=problem.xu, # search space: upper bounds
    max_iters=max_iters,
    verbose=True,
    metrics=metrics, # performance metrics 
    preconditioning=True, # perform preconditioning
)
X, Y, stopping_criteria = opt.run()
```
