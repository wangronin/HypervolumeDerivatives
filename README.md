# Hypervolume Indicator Derivatives

* **What for?**
  * a **Hypervolume Newton Method** for solving _continuous multiobjective optimization problems_ (MOPs),
  * enabled by the analytically computation of **Hypervolume Hessian Matrix**.
* **Why?**
  * the Newton method has a local **quadratic convergence** under some mild condition of objective functions.
  * Perhaps, you'd like to refine the final outcome of some direct optimizers with the Hypervolume Newton method..
* **When to use it?** the objective function is at least *twice continuously differentiable*.

![](assets/demo.png)

Specifically, you will find the following major functionalities:

1. `hvd.HypervolumeDerivatives`: the analytical computation of the HV Hessian and specifically, Alg. 2 described in [[DEW22]](https://arxiv.org/abs/2211.04171).
2. `hvd.HVN`: Hypervolume Newton Method for (Constrained) Multi-objective Optimization Problems in [[WED+22]](https://www.preprints.org/manuscript/202211.0103/v1).

## Installation

<!-- You could either install the stable version on `pypi`: -->
A `pypi` package will be available soon:
<!-- ```shell
pip install hvd
``` -->

For now, please take the lastest version from the master branch:

```shell
git clone https://github.com/wangronin/HypervolumeDerivatives.git
cd HypervolumeDerivatives && python setup.py install --user
```

## Hypervolume Hessian Matrix

Hypervolume (HV) Indicator of a point set $Y\subset\mathbb{R}^m$ computes the Lebesgue measure of the subset of $\mathbb{R}^m$ that is dominated by $Y$. HV is **Pareto compliant** and often used as a quality indicator in Evolutionary Multi-objective Optimization Algorithms (EMOAs), e.g., SMS-EMOA. Since maximizing HV w.r.t. the point set $S$ will lead to finite approximations to the (local) Pareto front, HV can also be used to guide the multi-objective search.

Consider an objective function $F:\mathbb{R}^d \rightarrow \mathbb{R}^m$, subject to minimization and a point set $X\subset \mathbb{R}^d$ of cardinality $n$. We care about the fast, analytical computation of the following quantity:
$$\frac{\partial^2 HV(F(X))}{\partial X \partial X^\top},$$
which is a $nd \times nd$-matrix. The implementation works for multi- and many-objective cases.

### Examples

Compute the HV Hessian w.r.t. the objective points, i.e., the objective function is an identity mapping:

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

Compute the HV Hessian w.r.t. the decision points. First, we define an objective function (the objective space is $\mathbb{R}^3$):

```Python
c1 = np.array([1.5, 0, np.sqrt(3) / 3])
c2 = np.array([1.5, 0.5, -1 * np.sqrt(3) / 6])
c3 = np.array([1.5, -0.5, -1 * np.sqrt(3) / 6])
ref = np.array([24, 24, 24])


def MOP1(x):
    x = np.array(x)
    return np.array(
        [
            np.sum((x - c1) ** 2),
            np.sum((x - c2) ** 2),
            np.sum((x - c3) ** 2),
        ]
    )

def MOP1_Jacobian(x):
    x = np.array(x)
    return np.array(
        [
            2 * (x - c1),
            2 * (x - c2),
            2 * (x - c3),
        ]
    )

def MOP1_Hessian(x):
    x = np.array(x)
    return np.array([2 * np.eye(3), 2 * np.eye(3), 2 * np.eye(3)])
```

Then, we compute the HV Hessian w.r.t. to the decision points in $X$:

```Python
hvh = HypervolumeDerivatives(
    n_decision_var=3, n_objective=3, ref=ref, func=MOP1, jac=MOP1_Jacobian, hessian=MOP1_Hessian
)

w = np.random.rand(20, 3)
w /= np.sum(w, axis=1).reshape(-1, 1)
X = w @ np.vstack([c1, c2, c3])
out = hvh.compute(X)
```

## Hypervolume Newton Method

The Hypervolume Newton Method is straightforward given the analytical computation of the HV Hessian:

$$X^{t+1} = X^{t} - \sigma\Bigg[\frac{\partial^2 HV(F(X))}{\partial X \partial X^\top}\Bigg]^{-1}X^{t}.$$

Note that, in the code base, we also implemented a HV Newton method for equality-constrained MOPs. 

### Example

We took the `MOP1` problem defined above:

```Python
from hvd.newton import HVN

max_iters = 30
mu = 20
ref = np.array([20, 20, 20])
w = np.abs(np.random.rand(mu, 3))
w /= np.sum(w, axis=1).reshape(-1, 1)
x0 = w @ np.vstack([c1, c2, c3])

opt = HVN(
    dim=2,
    n_objective=2,
    ref=ref,
    func=MOP1,
    jac=MOP1_Jacobian,
    hessian=MOP1_Hessian,
    mu=len(x0),
    x0=x0,
    lower_bounds=-2,
    upper_bounds=2,
    minimization=True,
    max_iters=max_iters,
    verbose=True,
)
X, Y, stop = opt.run()
```

## Brief Explanation of the Analytical Computation

The **hypervolume indicator** (HV) of a set of points is the m-dimensional Lebesgue measure of the space that is jointly dominated by a set of objective function vectors in $\mathbb{R}^m$ and bound from above by a reference point. HV is widely investigated in solving _multi-objective optimization problems_ (MOPs), where it is often used as a performance indicator for assessing the quality of _Evolutionay Multi-objective Optimization Algorithms_ (EMOAs), or employed to solve MOPs directly, e.g., [Hypervolume Indicator Gradient Algorithm](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Pz9c6XwAAAAJ&citation_for_view=Pz9c6XwAAAAJ:5nxA0vEk-isC) and [Hypervolume Indicator Netwon Method](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Pz9c6XwAAAAJ&citation_for_view=Pz9c6XwAAAAJ:QIV2ME_5wuYC).

We show an example of 3D hypervolume indicator and the geometrical meaning of its partial derivatives as follows.

![](assets/HV3D.png)

In this chart, we have three objective function to minimize, where we depicts three objective points, $y^{(i1)}, y^{(i2)}, y^{(i3)}$. The hypervolume (HV)indicator value, in this case, is the volume of the 3D ortho-convex polygon (in blue) - the subset dominated by $y^{(i1)}, y^{(i2)}, y^{(i3)}$. The first-order partial derivative of HV, for instance, $\partial HV/\partial y_3^{(i3)}$ is the yellow-colored 2D facet. The second-order partial derivative of HV, e.g., $\partial^2 HV/\partial y_3^{(i3)} \partial y_2^{(i2)}$ is an edge of the polygon.

## Symbolic computation of the Hessian in Mathematica

Also, we include, in folder `mathematica/`, several cases of the hypervolume indicator Hessian computed symoblically using `Mathematica`.

## References

* [[WED+22]](https://www.preprints.org/manuscript/202211.0103/v1) Wang, H.; Emmerich, Michael T. M.; Deutz, A.; Hernández, V.A.S.; Schütze, O. The Hypervolume Newton Method for Constrained Multi-objective Optimization Problems. _Preprints_ **2022**, 2022110103.

* [[DEW22]](https://arxiv.org/abs/2211.04171) Deutz, A.; Emmerich, Michael T. M.; Wang, H. The Hypervolume Indicator Hessian Matrix: Analytical Expression, Computational Time Complexity, and Sparsity, _arXiv_, 2022.










