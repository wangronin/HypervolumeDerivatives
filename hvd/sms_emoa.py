import numpy as np
from numba import jit
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.core.survival import Survival
from pymoo.indicators.hv.exact import DynamicHypervolume
from pymoo.indicators.hv.exact_2d import ExactHypervolume2D
from pymoo.indicators.hv.monte_carlo import ApproximateMonteCarloHypervolume
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize

from .exact import ExactHypervolume


@jit(nopython=True, error_model="numpy", cache=True)
def alpha(N, k):
    alpha = np.zeros(N+1)

    for i in range(1, N + 1):
        alpha[i] = np.prod(np.array([(k - j) / (N - j) for j in range(1, i)])) / i

    return alpha


@jit(nopython=True, error_model="numpy", cache=True)
def hv_monte_carlo(dom, V, n_dom=None):
    N, n_samples = dom.shape
    if n_dom is None:
        n_dom = np.sum(dom, axis=0)

    a = alpha(N, N)
    hv = np.sum(np.array([np.sum(a[n_dom[dom[i]]]) for i in range(N)])) / n_samples * V
    return hv

@jit(nopython=True, error_model="numpy", cache=True)
def hvc_monte_carlo(dom, V, n_dom=None, k=1):
    N, n_samples = dom.shape
    if n_dom is None:
        n_dom = np.sum(dom, axis=0)

    a = alpha(N, k)
    hvc = np.array([np.sum(np.array((np.sum(a[n_dom[dom[i]]]) / n_samples * V))) for i in range(N)])
    return hvc

class ApproximateMonteCarloHypervolume(DynamicHypervolume):

    def __init__(self, ref_point, n_samples=5000, n_exclusive=1, **kwargs) -> None:
        self.n_samples = n_samples
        self.n_exclusive = n_exclusive

        self.V = None
        self.dom = None

        super().__init__(ref_point, **kwargs)

    def _calc(self, ref_point, F):
        (N, M) = F.shape

        ideal = F.min(axis=0)
        V = np.prod(ref_point - ideal)

        S = np.random.uniform(low=ideal, high=ref_point, size=(self.n_samples, M))

        dom = np.array([np.all(F[i] <= S, axis=1) for i in range(N)])

        n_dom = dom.sum(axis=0)
        hv = hv_monte_carlo(dom, V, n_dom=n_dom)
        hvc = hvc_monte_carlo(dom, V, n_dom=n_dom, k=self.n_exclusive)

        self.V = V
        self.dom = dom

        return hv, hvc

    def delete(self, k):
        self.F = np.delete(self.F, k, axis=0)
        self.dom = np.delete(self.dom, k, axis=0)

        self.hv -= self.hvc[k]

        V, dom = self.V, self.dom
        n_dom = dom.sum(axis=0)
        self.hvc = hvc_monte_carlo(dom, V, n_dom=n_dom, k=self.n_exclusive)



class LeastHypervolumeContributionSurvival(Survival):
    def __init__(self, eps=10.0) -> None:
        super().__init__(filter_infeasible=True)
        self.eps = eps

    def _do(self, problem, pop, *args, n_survive=None, ideal=None, nadir=None, **kwargs):
        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # if the boundary points are not provided -> estimate them from pop
        if ideal is None:
            ideal = F.min(axis=0)
        if nadir is None:
            nadir = F.max(axis=0)

        # the number of objectives
        _, n_obj = F.shape

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):
            # get the actual front as individuals
            front = pop[front]
            front.set("rank", k)

            if len(survivors) + len(front) > n_survive:
                # normalize all the function values for the front
                F = front.get("F")
                F = normalize(F, ideal, nadir)

                # define the reference point and shift it a bit - F is already normalized!
                ref_point = np.full(problem.n_obj, 1.0 + self.eps)

                _, n_obj = F.shape

                # choose the suitable hypervolume method
                clazz = ExactHypervolume
                if n_obj == 2:
                    clazz = ExactHypervolume2D
                elif n_obj >= 3:
                    clazz = ApproximateMonteCarloHypervolume

                # finally do the computation
                hv = clazz(ref_point).add(F)

                # current front sorted by crowding distance if splitting
                while len(survivors) + len(front) > n_survive:
                    k = hv.hvc.argmin()
                    hv.delete(k)
                    front = np.delete(front, k)

            # extend the survivors by all or selected individuals
            survivors.extend(front)

        return Population.create(*survivors)


def cv_and_dom_tournament(pop, P, *args, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]
        (
            a_cv,
            a_f,
            b_cv,
            b_f,
        ) = (
            pop[a].CV[0],
            pop[a].F,
            pop[b].CV[0],
            pop[b].F,
        )

        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method="smaller_is_better", return_random_if_equal=True)

        # both solutions are feasible
        else:
            # if one dominates another choose the nds one
            rel = Dominator.get_relation(a_f, b_f)
            if rel == 1:
                S[i] = a
            elif rel == -1:
                S[i] = b

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = np.random.choice([a, b])

    return S[:, None].astype(int, copy=False)


class SMSEMOA(GeneticAlgorithm):
    def __init__(
        self,
        pop_size=100,
        sampling=FloatRandomSampling(),
        selection=TournamentSelection(func_comp=cv_and_dom_tournament),
        crossover=SBX(),
        mutation=PM(),
        survival=LeastHypervolumeContributionSurvival(),
        eliminate_duplicates=True,
        n_offsprings=None,
        normalize=True,
        output=MultiObjectiveOutput(),
        **kwargs
    ):
        """

        Parameters
        ----------
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            eliminate_duplicates=eliminate_duplicates,
            n_offsprings=n_offsprings,
            output=output,
            advance_after_initial_infill=True,
            **kwargs
        )

        self.normalize = normalize

    def _advance(self, infills=None, **kwargs):
        ideal, nadir = None, None

        # estimate ideal and nadir from the current population (more robust then from doing it from merged)
        if self.normalize:
            F = self.pop.get("F")
            ideal, nadir = F.min(axis=0), F.max(axis=0) + 1e-32

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)

        self.pop = self.survival.do(
            self.problem, pop, n_survive=self.pop_size, algorithm=self, ideal=ideal, nadir=nadir, **kwargs
        )
