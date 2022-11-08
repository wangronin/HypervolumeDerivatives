# Hypervolume Indicator Derivatives

### Python package to compute the Hessian matrix of hypervolume indicator analytically

This packakge contains `Python` implementations of the algorithms in the following two papers:

[WED+22] Wang, H.; Emmerich, M.; Deutz, A.; Hernández, V.A.S.; Schütze, O. The Hypervolume Newton Method for Constrained Multi-objective Optimization Problems. _Preprints_ **2022**, 2022110103. [[PDF]](https://www.preprints.org/manuscript/202211.0103/v1)

[DEW22] TBA

Specifically, you will find the following major functionalities:

1. the analytical computation of the Hessian and specifically, Alg. 2 described in [DEW22]: module `hvd.HypervolumeDerivatives`
2. Hypervolume Newton Method for Constrained Multi-objective Optimization Problems in [WED+22]: module `hvd.HVN`

The **hypervolume indicator** (HV) of a set of points is the m-dimensional Lebesgue measure of the space that is jointly dominated by a set of objective function vectors in R^m and bound from above by a reference point. HV is widely investigated in solving _multi-objective optimization problems_ (MOPs), where it is often used as a performance indicator for assessing the quality of _Evolutionay Multi-objective Optimization Algorithms_ (EMOAs), or employed to solve MOPs directly, e.g., [Hypervolume Indicator Gradient Algorithm](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Pz9c6XwAAAAJ&citation_for_view=Pz9c6XwAAAAJ:5nxA0vEk-isC) and [Hypervolume Indicator Netwon Method](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Pz9c6XwAAAAJ&citation_for_view=Pz9c6XwAAAAJ:QIV2ME_5wuYC).

We show an example of 3D hypervolume indicator and the geometrical meaning of its partial derivatives as follows.

![](assets/HV3D.png)

### Installation

You could either install the stable version on `pypi`:

```shell
pip install hvd
```

Or, take the lastest version from the master branch:

```shell
git clone https://github.com/wangronin/HypervolumeDerivatives.git
cd HypervolumeDerivatives && python setup.py install --user
```

### Example

```Python
from hvd import HypervolumeDerivatives

ref = np.array([9, 10, 12])
hvh = HypervolumeDerivatives(3, 3, ref, minimization=True)
out = hvh.compute_hessian(X=np.array([[5, 3, 7], [2, 1, 10]]))

# out["HVdY2"] = np.array(
#     [
#         [0, 3, 7, 0, 0, -7],
#         [3, 0, 4, 0, 0, -4],
#         [7, 4, 0, 0, 0, 0],
#         [0, -0, 0, 0, 2, 9],
#         [-0, 0, 0, 2, 0, 7],
#         [-7, -4, 0, 9, 7, 0],
#     ]
# )
```

### Background on Multi-objective optimization

Continuous m-dimensional multi-objective optimization problems (MOPs), where multiple objective functions, e.g., $\mathbf{f}=(f_1, \ldots, f_m): \mathbb{R}^d \rightarrow \mathbb{R}^m$ are subject to minimization. Also, we assume $\mathbf{f}$ is at least twice continuously differentiable. When solving such problems, it is a common strategy to approximate the Pareto front for $m$-objective functions mapping from a continuous decision space $\mathbb{R}^d$ to the $\mathbb{R}$ (or as a vector-valued function from $\mathbb{R}^d$ to $\mathbb{R}^m$. MOPs can be accomplished by means of a finite set of points that distributes across the at most $m-1$-dimensional manifold of the Pareto front.







