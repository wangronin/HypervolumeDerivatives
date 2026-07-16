import numpy as np
from pymoo.algorithms.moo.moead import MOEAD, default_decomp, default_ref_dirs
from pymoo.util.misc import parameter_less
from scipy.spatial.distance import cdist


class ConstraintAwareMOEAD(MOEAD):
    """MOEAD with local adaptive-epsilon constraint handling.

    Pymoo 0.6.1.1's AdaptiveEpsilonConstraintHandling is not compatible with
    loopwise MOEAD: it skips MOEAD._setup and assumes Population infills.
    """

    def __init__(self, *args, perc_eps_until=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.perc_eps_until = perc_eps_until
        self.max_cv = None

    def next(self):
        if not self.is_initialized:
            return super().next()

        # Stock MOEAD yields one offspring per call; drain it to one generation per call.
        n_gen = self.n_gen
        while self.has_next() and self.n_gen == n_gen:
            super().next()

    def _setup(self, problem, **kwargs):
        if self.ref_dirs is None:
            self.ref_dirs = default_ref_dirs(problem.n_obj)
        self.pop_size = len(self.ref_dirs)
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind="quicksort")[
            :, : self.n_neighbors
        ]
        if self.decomposition is None:
            self.decomposition = default_decomp(problem)

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills=infills, **kwargs)
        if self.problem.has_constraints() and infills is not None:
            self.max_cv = float(np.mean(infills.get("CV")))

    def _cv_eps(self):
        if self.max_cv is None:
            return 0.0
        t = self.termination.perc if self.termination is not None else 1.0
        alpha = max(0.0, 1 - 1 / self.perc_eps_until * t)
        return alpha * self.max_cv

    def _replace(self, k, off):
        pop = self.pop
        N = self.neighbors[k]
        FV = self.decomposition.do(pop[N].get("F"), weights=self.ref_dirs[N, :], ideal_point=self.ideal)
        off_FV = self.decomposition.do(off.F[None, :], weights=self.ref_dirs[N, :], ideal_point=self.ideal)

        if self.problem.has_constraints():
            eps = self._cv_eps()
            CV = np.maximum(pop[N].get("CV")[:, 0] - eps, 0.0)
            off_CV = np.maximum(np.full(len(off_FV), off.CV[0]) - eps, 0.0)
            fmax = max(FV.max(), off_FV.max())
            FV = parameter_less(FV, CV, fmax=fmax)
            off_FV = parameter_less(off_FV, off_CV, fmax=fmax)

        I = np.where(off_FV < FV)[0]
        pop[N[I]] = off
