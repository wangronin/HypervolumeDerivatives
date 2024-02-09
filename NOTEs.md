# ZDTs

## gen = 1510

* ZDT6 requires Hessian preconditioning and a large initial shift of the reference set (0.3 * `n`)
* ZDT3 has a few outliers, which could be caused by ill-conditioned Hessian of IGD
* On ZDT1-4, a small initial shift (0.01 * `n`) leads to better results


## DTLZs

* DTLZ6 is non-differentiable at the efficient set.

## CFs

* box-constraint handling is needed here: projecting the Newton's direction on the box boundary, at least on CF5.
* Converting box constraints to inequality won't work since sometimes the Newton's direction can go too far into the infeasible region, causing the Jacobian/Hessian to diverge.
