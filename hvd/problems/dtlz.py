import jax.numpy as jnp
from pymoo.core.problem import Problem
from pymoo.problems.many.dtlz import get_ref_dirs
from pymoo.util.remote import Remote


class DTLZ(Problem):
    def __init__(self, n_var, n_obj, k=None, **kwargs):
        if n_var:
            self.k = n_var - n_obj + 1
        elif k:
            self.k = k
            n_var = k + n_obj - 1
        else:
            raise Exception("Either provide number of variables or k!")

        super().__init__(n_var=n_var, n_obj=n_obj, xl=0, xu=1, vtype=float, **kwargs)

    def g1(self, X_M):
        return 100 * (self.k + jnp.sum(jnp.square(X_M - 0.5) - jnp.cos(20 * jnp.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        return jnp.sum(jnp.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = 1 + g
            _f *= jnp.prod(jnp.cos(jnp.power(X_[:, : X_.shape[1] - i], alpha) * jnp.pi / 2.0), axis=1)
            if i > 0:
                _f *= jnp.sin(jnp.power(X_[:, X_.shape[1] - i], alpha) * jnp.pi / 2.0)

            f.append(_f)

        f = jnp.column_stack(f)
        return f


class DTLZ1(DTLZ):
    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs=None):
        if ref_dirs is None:
            ref_dirs = get_ref_dirs(self.n_obj)
        return 0.5 * ref_dirs

    def obj_func(self, X_, g):
        f = []

        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= jnp.prod(X_[:, : X_.shape[1] - i], axis=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)

        return jnp.column_stack(f)

    def _evaluate(self, x):
        x = jnp.array([x])
        X_, X_M = x[:, : self.n_obj - 1], x[:, self.n_obj - 1 :]
        g = self.g1(X_M)
        return self.obj_func(X_, g)[0]


class DTLZ7(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)

    def _calc_pareto_front(self):
        if self.n_obj == 3:
            return Remote.get_instance().load("pymoo", "pf", "dtlz7-3d.pf")
        else:
            raise Exception("Not implemented yet.")

    def _evaluate(self, x):
        x = jnp.array([x])
        f = []
        for i in range(0, self.n_obj - 1):
            f.append(x[:, i])
        f = jnp.column_stack(f)

        g = 1 + 9 / self.k * jnp.sum(x[:, -self.k :], axis=1)
        h = self.n_obj - jnp.sum(f / (1 + g[:, None]) * (1 + jnp.sin(3 * jnp.pi * f)), axis=1)

        return jnp.column_stack([f, (1 + g) * h])[0]
