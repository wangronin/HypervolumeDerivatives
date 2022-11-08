# Hypervolume Indicator Hessian Matrix

### Python package to compute the Hessian matrix of hypervolume indicator analytically

This packakge contains `Python` implementations of the analytical computation and speficically Alg. 2 described in the following paper:

The **hypervolume indicator** (HV) of a set of points is the m-dimensional Lebesgue measure of the space that is jointly dominated by a set of objective function vectors in R^m and bound from above by a reference point. HV is widely used as performance indicator for assessing the quality of _Evolutionay Multi-objective Optimization Algorithms_ (EMOAs), or as a

We show an example of 3D hypervolume indicator and the geometrical meaning of its partial derivatives as follows.

![](assets/HV3D.png)

## Installation

You could either install the stable version on `pypi`:

```shell
pip install hvd
```

Or, take the lastest version from the master branch:

```shell
git clone https://github.com/wangronin/HypervolumeDerivatives.git
cd HypervolumeDerivatives && python setup.py install --user
```


### Background on Multi-objective optimization

Continuous m-dimensional multi-objective optimization problems (MOPs), where multiple
objective functions, e.g., f = (f1, . . . , fm) : X ⊆ R^d → R^m are subject to minimization. Also, we assume f is at least
twice continuously differentiable. When solving such problems, it is a common strategy to approximate the Pareto
front for m-objective functions mapping from a continuous decision space R^d to R






