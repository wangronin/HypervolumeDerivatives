import functools
import logging
import statistics
import sys
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import cholesky, qr
from sklearn.neighbors import LocalOutlierFactor

__author__ = "Hao Wang"


# def _handle_box_constraint(self, step: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     if 1 < 2:
#         return step, np.ones(len(step))

#     primal_vars = self.state.primal
#     step_primal = step[:, : self.dim_p]
#     normal_vectors = np.c_[np.eye(self.dim_p), -1 * np.eye(self.dim_p)]
#     # calculate the maximal step-size
#     dist = np.c_[
#         np.abs(primal_vars - self.xl),
#         np.abs(self.xu - primal_vars),
#     ]
#     v = step_primal @ normal_vectors
#     s = np.array([dist[i] / np.abs(np.minimum(0, vv)) for i, vv in enumerate(v)])
#     max_step_size = np.array([min(1.0, np.nanmin(_)) for _ in s])
#     # project Newton's direction onto the box boundary
#     idx = max_step_size == 0
#     if np.any(idx) > 0:
#         proj_dim = [np.argmin(_) for _ in s[idx]]
#         proj_axis = normal_vectors[:, proj_dim]
#         step_primal[idx] -= (np.einsum("ij,ji->i", step_primal[idx], proj_axis) * proj_axis).T
#         step[:, : self.dim_p] = step_primal
#         # re-calculate the `max_step_size` for projected directions
#         v = step[:, : self.dim_p] @ normal_vectors
#         s = np.array([dist[i] / np.abs(np.minimum(0, vv)) for i, vv in enumerate(v)])
#         max_step_size = np.array([min(1, np.nanmin(_)) for _ in s])
#     return step, max_step_size


def merge_lists(x, y):
    if x is None:
        return y
    if y is None:
        return x

    assert len(x) == len(y)
    out = []
    for k in range(len(x)):
        v, w = x[k], y[k]
        N = len(v)
        out.append(np.array([np.r_[v[i], w[i]] for i in range(N)]))
    return tuple(out)


def compute_chim(Y: np.ndarray) -> np.ndarray:
    # compute the normal vector to CHIM
    idx = Y.argmin(axis=0)
    Z = Y[idx, :]
    M = Z[1:] - Z[0]
    Q = qr(M.T)[0]
    n = -1 * np.abs(Q[:, -1])
    n /= np.linalg.norm(n)
    return n.ravel()


def precondition_hessian(H: np.ndarray) -> np.ndarray:
    """Precondition the Hessian matrix to make sure it is positive definite

    Args:
        H (np.ndarray): the Hessian matrix

    Returns:
        np.ndarray: the lower triagular decomposition of the preconditioned Hessian
    """
    # pre-condition the Hessian
    try:
        L = cholesky(H, lower=True)
    except:
        beta = 1e-6
        v = np.min(np.diag(H))
        tau = 0 if v > 0 else -v + beta
        I = np.eye(H.shape[0])
        for _ in range(35):
            try:
                L = cholesky(H + tau * I, lower=True)
                break
            except:
                tau = max(2 * tau, beta)
        else:
            print("Pre-conditioning the HV Hessian failed")
    return L @ (L.T)


def preprocess_reference_set(X: np.ndarray) -> np.ndarray:
    """remove duplicated points and outliers"""
    N = len(X)
    idx = []
    # remove duplicated points
    for i in range(N):
        x = X[i]
        CON = np.all(
            np.isclose(X[np.arange(N) != i], x, rtol=1e-3, atol=1e-4),
            axis=1,
        )
        if all(~CON):
            idx.append(i)

    if len(idx) > 50:
        X = X[idx]
    score = LocalOutlierFactor(n_neighbors=max(5, int(len(X) / 10))).fit_predict(X)
    return X[score != -1]


class LoggerFormatter(logging.Formatter):
    """TODO: use relative path for %(pathname)s"""

    default_time_format = "%m/%d/%Y %H:%M:%S"
    default_msec_format = "%s,%02d"
    FORMATS = {
        logging.DEBUG: ("%(asctime)s - [%(name)s.%(levelname)s] {%(pathname)s:%(lineno)d} -- %(message)s"),
        logging.INFO: "%(asctime)s - [%(name)s.%(levelname)s] -- %(message)s",
        logging.WARNING: "%(asctime)s - [%(name)s.%(levelname)s] {%(name)s} -- %(message)s",
        logging.ERROR: ("%(asctime)s - [%(name)s.%(levelname)s] {%(pathname)s:%(lineno)d} -- %(message)s"),
        "DEFAULT": "%(asctime)s - %(levelname)s -- %(message)s",
    }

    def __init__(self, fmt="%(asctime)s - %(levelname)s -- %(message)s"):
        LoggerFormatter.FORMATS["DEFAULT"] = fmt
        super().__init__(fmt=fmt, datefmt=None, style="%")

    def format(self, record):
        # Save the original format configured by the user
        # when the logger formatter was instantiated
        _fmt = getattr(self._style, "_fmt")
        # Replace the original format with one customized by logging level
        setattr(self._style, "_fmt", self.FORMATS.get(record.levelno, self.FORMATS["DEFAULT"]))
        # Call the original formatter class to do the grunt work
        fmt = logging.Formatter.format(self, record)
        # Restore the original format configured by the user
        setattr(self._style, "_fmt", _fmt)
        return fmt


def get_logger(logger_id: str, file: Union[str, List[str]] = None, console: bool = False) -> logging.Logger:
    # NOTE: logging.getLogger create new instance based on `name`
    # no new instance will be created if the same name is provided
    logger = logging.getLogger(str(logger_id))
    logger.setLevel(logging.DEBUG)

    fmt = LoggerFormatter()
    # create console handler and set level to the vebosity
    SH = list(filter(lambda h: isinstance(h, logging.StreamHandler), logger.handlers))
    if len(SH) == 0 and console:  # if console handler is not registered yet
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    # create file handler and set level to debug
    FH = list(filter(lambda h: isinstance(h, logging.FileHandler), logger.handlers))
    if file is not None and len(FH) == 0:
        file = [file] if isinstance(file, str) else file
        for f in set(file) - set(fh.baseFilename for fh in FH):
            try:
                fh = logging.FileHandler(f)
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(fmt)
                logger.addHandler(fh)
            except FileNotFoundError as _:
                pass

    logger.propagate = False
    return logger


def timeit(func):
    @functools.wraps(func)
    def __func__(ref, *arg, **kwargv):
        t0 = time.process_time_ns()
        out = func(ref, *arg, **kwargv)
        t1 = time.process_time_ns()
        ref.CPU_time += t1 - t0
        return out

    return __func__


# TODO: this is too slow, improve the algorithm
# @jit(nopython=True, error_model="numpy", cache=True)
def get_non_dominated(pareto_front: np.ndarray, return_index: bool = False, weakly_dominated: bool = True):
    """Find pareto front (undominated part) of the input performance data.
    Minimization is assumed

    """
    pareto_indices = []
    for idx, p in enumerate(pareto_front):
        if weakly_dominated:
            cond = np.all(np.any(pareto_front[:idx] > p, axis=1)) and np.all(
                np.any(pareto_front[idx + 1 :] > p, axis=1)
            )
        else:
            cond = np.all(
                np.any(pareto_front[:idx] > p, axis=1) & np.all(~np.isclose(pareto_front[:idx], p), axis=1)
            ) and np.all(
                np.any(pareto_front[idx + 1 :] > p, axis=1)
                & np.all(~np.isclose(pareto_front[idx + 1 :], p), axis=1)
            )
        if cond:
            pareto_indices.append(idx)
    pareto_indices = np.array(pareto_indices)
    pareto_front = pareto_front[pareto_indices].copy()
    return pareto_indices if return_index else pareto_front


def set_bounds(bound, dim):
    if isinstance(bound, str):
        bound = eval(bound)
    elif isinstance(bound, (float, int)):
        bound = [bound] * dim
    elif hasattr(bound, "__iter__"):
        bound = list(bound)
        if len(bound) == 1:
            bound *= dim
    assert len(bound) == dim
    return np.asarray(bound)


def handle_box_constraint(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """This function transforms x to t w.r.t. the low and high
    boundaries lb and ub. It implements the function T^{r}_{[a,b]} as
    described in Rui Li's PhD thesis "Mixed-Integer Evolution Strategies
    for Parameter Optimization and Their Applications to Medical Image
    Analysis" as alorithm 6.

    """
    x = np.asarray(x, dtype="float")
    shape_ori = x.shape
    x = np.atleast_2d(x)
    lb = np.atleast_1d(lb)
    ub = np.atleast_1d(ub)

    transpose = False
    if x.shape[0] != len(lb):
        x = x.T
        transpose = True

    lb, ub = lb.flatten(), ub.flatten()
    lb_index = np.isfinite(lb)
    up_index = np.isfinite(ub)

    valid = np.bitwise_and(lb_index, up_index)

    LB = lb[valid][:, np.newaxis]
    UB = ub[valid][:, np.newaxis]

    y = (x[valid, :] - LB) / (UB - LB)
    I = np.mod(np.floor(y), 2) == 0
    yprime = np.zeros(y.shape)
    yprime[I] = np.abs(y[I] - np.floor(y[I]))
    yprime[~I] = 1.0 - np.abs(y[~I] - np.floor(y[~I]))

    x[valid, :] = LB + (UB - LB) * yprime

    if transpose:
        x = x.T
    return x.reshape(shape_ori)


def is_dominate(leftv: Sequence[Any], rightv: Sequence[Any]) -> bool:
    """Check. Does a 'leftv' dominate a 'rightv'?
    A 'leftv' dominates a 'rightv', if and only if leftv[i] <= rightv[i], for all i in {0,1,..., len(leftv) - 1},
    and there exists j in {0,1,...,len(leftv) - 1}: leftv[j] < rightv[j].
    --------------------
    Args:
        'leftv': A first vector of the values.
        'rightv': A second vector of the values.
    --------------------
    Returns:
        True if 'leftv' dominates a 'rightv', otherwise False.
    """

    assert len(leftv) == len(rightv), "'leftv' must have a same length as 'rightv'."

    is_all_values_less_or_eq = True
    is_one_value_less = False

    for i in range(len(leftv)):
        if leftv[i] < rightv[i]:
            is_one_value_less = True
        elif leftv[i] > rightv[i]:
            is_all_values_less_or_eq = False
            break
    return is_all_values_less_or_eq and is_one_value_less


def _is_seq_has_one_uniq_value(iterable: Iterable[Any]) -> bool:
    """Check. Has 'iterable' only a one unique value?

    It is equivalent following: 'len({item for item in iterable}) == 1'.

    --------------------
    Args:
        'iterable': An input sequence.

    --------------------
    Returns:
        True, if 'iterable' contains only a one unique value, otherwise False.

    --------------------
    Raises:
        ValueError: If 'iterable' is empty.

    """

    iterator = iter(iterable)

    try:
        first_value = next(iterator)
        is_has_uniq_value = True
    except StopIteration:
        raise ValueError("'iterable' is empty.")

    try:
        while True:
            value = next(iterator)
            if value != first_value:
                is_has_uniq_value = False
                break
    except StopIteration:
        pass

    return is_has_uniq_value


def _merge(indices1: List[int], indices2: List[int]) -> List[int]:
    """Merge the two list of the indices. Each list must be sorted.

    --------------------
    Args:
        'indices1': A sorted list of the indices.
        'indices2': A sorted list of the indices.

    --------------------
    Returns:
        The ordered list of indices.

    """
    merged_list = indices1 + indices2

    index1 = 0
    index2 = 0

    merge_index = 0

    while index1 < len(indices1) and index2 < len(indices2):
        if indices1[index1] < indices2[index2]:
            merged_list[merge_index] = indices1[index1]
            index1 += 1
        else:
            merged_list[merge_index] = indices2[index2]
            index2 += 1
        merge_index += 1

    for i in range(index1, len(indices1)):
        merged_list[merge_index] = indices1[i]
        merge_index += 1

    for i in range(index2, len(indices2)):
        merged_list[merge_index] = indices2[i]
        merge_index += 1

    return merged_list


def _split_by(
    seq_objs_front: List[Dict[str, Union[int, Any]]], indices: List[int], split_value: Any, index_value: int
) -> Tuple[List[int], List[int], List[int]]:
    """'indices' splits into three lists.

    The three lits are the list of indices, where 'index_value'th value of the objectives is less than a 'split_value',
    the list of indices, where 'index_value'th value of the objectives is equal to a 'split_value',
    the list of indices, where 'index_value'th value of the objectives is greater than a 'split_value'.

    --------------------
    Args:
         'seq_objs_front': A dictionary contains the values of the objectives and indices of the fronts.
         'indices': The indices of the objectives in the 'seq_objs_front'.
         'split_value': A value for the splitting.
         'index_value': The index of the value in the objectives, for the split.

    --------------------
    Returns:
         The tuple of lists of the indices.

    """
    indices_less_split_value = []
    indices_greater_split_value = []
    indices_equal_split_value = []

    for index in indices:
        if seq_objs_front[index]["objs"][index_value] < split_value:
            indices_less_split_value.append(index)
        elif seq_objs_front[index]["objs"][index_value] > split_value:
            indices_greater_split_value.append(index)
        else:
            indices_equal_split_value.append(index)

    return indices_less_split_value, indices_equal_split_value, indices_greater_split_value


def _sweep_a(seq_objs_front: List[Dict[str, Union[Any, str]]], indices: List[int]) -> None:
    """Two-objective sorting.

    It attributes front's index to the lexicographically ordered elements in the  'seq_objs_front',
    with the indices in the 'indices', based on the first two values of the objectives using a line-sweep algorithm.

    --------------------
    Args:
        'seq_objs_front': A dictionary contains the values of the objectives and indices of the fronts.
        'indices': The indices of the objectives in the 'seq_objs_front'.

    --------------------
    Returns:
        None

    """
    init_ind = set((indices[0],))

    for k in range(1, len(indices)):
        i = indices[k]
        indices_where_sec_values_less_or_eq = [
            index for index in init_ind if seq_objs_front[index]["objs"][1] <= seq_objs_front[i]["objs"][1]
        ]
        if indices_where_sec_values_less_or_eq:
            max_front = max(seq_objs_front[index]["front"] for index in indices_where_sec_values_less_or_eq)
            seq_objs_front[i]["front"] = max(seq_objs_front[i]["front"], max_front + 1)

        init_ind -= {
            index for index in init_ind if seq_objs_front[index]["front"] == seq_objs_front[i]["front"]
        }
        init_ind.add(i)


def _sweep_b(
    seq_objs_front: List[Dict[str, Union[Any, int]]], comp_indices: List[int], assign_indices: List[int]
) -> None:
    """Two-objective sorting procedure.

    It attributes front's indices to elements in the 'seq_objs_front', with the indices in the 'assign_indices',
    based on the first two values of the objectives by comparing them to fitnesses,
    with the indices in the  'comp_indices', using a line-sweep algorithm.

    --------------------
    Args:
        'seq_objs_front': A dictionary contains the values of the objectives and indices of the fronts.
        'comp_indices': The indices for comparing.
        'assign_indices': The indices for assign front.

    --------------------
    Returns:
        None

    """

    init_ind = set()
    p = 0

    for j in assign_indices:
        if p < len(comp_indices):
            fitness_right = seq_objs_front[j]["objs"][:2]

        while p < len(comp_indices):
            i = comp_indices[p]
            fitness_left = seq_objs_front[i]["objs"][:2]
            if fitness_left <= fitness_right:
                indices_less_value_eq_front = [
                    index
                    for index in init_ind
                    if seq_objs_front[index]["front"] == seq_objs_front[i]["front"]
                    and seq_objs_front[index]["objs"][1] < seq_objs_front[i]["objs"][1]
                ]

                if not indices_less_value_eq_front:
                    init_ind -= {
                        index
                        for index in init_ind
                        if seq_objs_front[index]["front"] == seq_objs_front[i]["front"]
                    }
                    init_ind.add(i)
                p += 1
            else:
                break
        indices_less_or_eq_value = [
            index for index in init_ind if seq_objs_front[index]["objs"][1] <= seq_objs_front[j]["objs"][1]
        ]

        if indices_less_or_eq_value:
            max_front = max(seq_objs_front[index]["front"] for index in indices_less_or_eq_value)
            seq_objs_front[j]["front"] = max(seq_objs_front[j]["front"], max_front + 1)


def _nd_helper_a(
    seq_objs_front: List[Dict[str, Union[Any, int]]], indices: List[int], count_of_obj: int
) -> None:
    """Recursive procedure.

    It attributes front's indices to all elements in the 'seq_objs_front', with the indices in the 'indices',
    for the first 'count_of_obj' values of the objectives.

    --------------------
    Args:
         'seq_objs_front': A dictionary contains the values of the objectives and indices of the fronts.
         'indices': The indices for assign front.
         'count_of_obj': The number of the values from the objectives, for the sorting.

    --------------------
    Returns:
         None

    """

    if len(indices) < 2:
        return
    elif len(indices) == 2:
        index_l, index_r = indices[0], indices[1]
        fitness1, fitness2 = (
            seq_objs_front[index_l]["objs"][:count_of_obj],
            seq_objs_front[index_r]["objs"][:count_of_obj],
        )

        if is_dominate(fitness1, fitness2):
            seq_objs_front[index_r]["front"] = max(
                seq_objs_front[index_r]["front"], seq_objs_front[index_l]["front"] + 1
            )
    elif count_of_obj == 2:
        _sweep_a(seq_objs_front, indices)
    elif _is_seq_has_one_uniq_value(seq_objs_front[index]["objs"][count_of_obj - 1] for index in indices):
        _nd_helper_a(seq_objs_front, indices, count_of_obj - 1)
    else:
        median = statistics.median_low(seq_objs_front[index]["objs"][count_of_obj - 1] for index in indices)

        less_median, equal_median, greater_median = _split_by(
            seq_objs_front, indices, median, count_of_obj - 1
        )

        less_and_equal_median = _merge(equal_median, less_median)

        _nd_helper_a(seq_objs_front, less_median, count_of_obj)
        _nd_helper_b(seq_objs_front, less_median, equal_median, count_of_obj - 1)
        _nd_helper_a(seq_objs_front, equal_median, count_of_obj - 1)
        _nd_helper_b(seq_objs_front, less_and_equal_median, greater_median, count_of_obj - 1)
        _nd_helper_a(seq_objs_front, greater_median, count_of_obj)


def _nd_helper_b(
    seq_objs_front: List[Dict[str, Union[Any, int]]],
    comp_indices: List[int],
    assign_indices: List[int],
    count_of_obj: int,
) -> None:
    """Recursive procedure.

    It attributes a front's indices to all elements in the 'seq_objs_front', with the indices in the  'assign_indices',
    for the first 'count_of_obj' values of the objectives, by comparing them to elements in the 'seq_objs_front',
    with the indices in the 'comp_indices'.

    --------------------
    Args:
         'seq_objs_front': A dictionary contains the values of the objectives and indices of the fronts.
         'comp_indices': The indices for comparing.
         'assign_indices': The indices for assign front.
         'count_of_obj': The number of the values from the objectives, for the sorting.

    --------------------
    Returns:
         None

    """

    if not comp_indices or not assign_indices:
        return
    elif len(comp_indices) == 1 or len(assign_indices) == 1:
        for i in assign_indices:
            hv = seq_objs_front[i]["objs"][:count_of_obj]
            for j in comp_indices:
                lv = seq_objs_front[j]["objs"][:count_of_obj]
                if is_dominate(lv, hv) or lv == hv:
                    seq_objs_front[i]["front"] = max(
                        seq_objs_front[i]["front"], seq_objs_front[j]["front"] + 1
                    )
    elif count_of_obj == 2:
        _sweep_b(seq_objs_front, comp_indices, assign_indices)
    else:
        values_objs_from_comp_indices = {seq_objs_front[i]["objs"][count_of_obj - 1] for i in comp_indices}
        values_objs_from_assign_indices = {
            seq_objs_front[j]["objs"][count_of_obj - 1] for j in assign_indices
        }

        min_from_comp_indices, max_from_comp_indices = min(values_objs_from_comp_indices), max(
            values_objs_from_comp_indices
        )

        min_from_assign_indices, max_from_assign_indices = min(values_objs_from_assign_indices), max(
            values_objs_from_assign_indices
        )

        if max_from_comp_indices <= min_from_assign_indices:
            _nd_helper_b(seq_objs_front, comp_indices, assign_indices, count_of_obj - 1)
        elif min_from_comp_indices <= max_from_assign_indices:
            median = statistics.median_low(values_objs_from_comp_indices | values_objs_from_assign_indices)

            less_median_indices_1, equal_median_indices_1, greater_median_indices_1 = _split_by(
                seq_objs_front, comp_indices, median, count_of_obj - 1
            )
            less_median_indices_2, equal_median_indices_2, greater_median_indices_2 = _split_by(
                seq_objs_front, assign_indices, median, count_of_obj - 1
            )

            less_end_equal_median_indices_1 = _merge(less_median_indices_1, equal_median_indices_1)

            _nd_helper_b(seq_objs_front, less_median_indices_1, less_median_indices_2, count_of_obj)
            _nd_helper_b(seq_objs_front, less_median_indices_1, equal_median_indices_2, count_of_obj - 1)
            _nd_helper_b(seq_objs_front, equal_median_indices_1, equal_median_indices_2, count_of_obj - 1)
            _nd_helper_b(
                seq_objs_front, less_end_equal_median_indices_1, greater_median_indices_2, count_of_obj - 1
            )
            _nd_helper_b(seq_objs_front, greater_median_indices_1, greater_median_indices_2, count_of_obj)


def non_domin_sort(
    decisions: Iterable[Any],
    get_objectives: Callable[[Any], Iterable[Any]] = None,
    only_front_indices: bool = False,
) -> Union[Tuple[int], Dict[int, Tuple[Any]]]:
    """A non-dominated sorting.

    If 'get_objectives' is 'None', then it is identity map: 'get_objectives = lambda x: x'.

    --------------------
    Args:
        'decisions': The sequence of the decisions for non-dominated sorting.
        'get_objectives': The function which maps a decision space into a objectives space.
        'only_front_indices':

    --------------------
    Returns:
        If 'only_front_indices' is False:
            A dictionary. It contains indices of fronts as keys and values are tuple consist of
            'decisions' which have a same index of the front.
        Otherwise:
            Tuple of front's indices for the every decision in 'decisions'.
    """

    # The dictionary contains the objectives as keys and indices of the their preimages in the 'decisions' as values.
    objs_dict = defaultdict(list)

    if get_objectives is None:
        objs_gen = map(lambda x: (x, tuple(x)), decisions)
    else:
        objs_gen = map(lambda x: (x, tuple(get_objectives(x))), decisions)

    for index, (decision, fitness) in enumerate(objs_gen):
        objs_dict[fitness].append((index, decision))

    total_unique_objs = 0

    for objs in objs_dict:
        if total_unique_objs == 0:
            first_obj = objs
            count_of_obj = len(objs)
            assert count_of_obj > 1, (
                "The number of the objectives must be > 1, "
                "but image of the decision have the length is {0}."
                "\nThe indices of the decisions: {1}.".format(
                    count_of_obj, [index for (index, dec) in objs_dict[objs]]
                )
            )
        else:
            assert count_of_obj == len(objs), (
                "The images of the decisions at positions {0} "
                "have the number of the objectives "
                "is not equal the number of the objectives of the images at positions "
                "{1}.".format(
                    [index for (index, dec) in objs_dict[first_obj]],
                    [index for (index, dec) in objs_dict[objs]],
                )
            )
        total_unique_objs += 1

    assert total_unique_objs != 0, "The sequence of the decisions or values of the objectives is empty."

    # The list 'unique_objs' never changes, but its elements yes.
    # It sorted in the lexicographical order.
    unique_objs_and_fronts = [{"objs": fitness, "front": 0} for fitness in sorted(objs_dict.keys())]

    # Further, algorithm works only with the indices of list 'unique_objs'.
    indices_uniq_objs = list(range(len(unique_objs_and_fronts)))
    _nd_helper_a(unique_objs_and_fronts, indices_uniq_objs, count_of_obj)

    if only_front_indices is True:
        total_decisions = sum(map(len, (objs_dict[objs] for objs in objs_dict)))
        fronts = list(range(total_decisions))
        for objs in unique_objs_and_fronts:
            for index, dec in objs_dict[objs["objs"]]:
                fronts[index] = objs["front"]
        fronts = np.array(tuple(fronts))
        fronts = {i: np.nonzero(fronts == i)[0] for i in np.unique(fronts)}
    else:
        # The dictionary contains indices of the fronts as keys and the tuple of 'decisions' as values.
        fronts = defaultdict(tuple)

        # Generate fronts.
        for objs_front in unique_objs_and_fronts:
            fronts[objs_front["front"]] += tuple(
                decision for (index, decision) in objs_dict[objs_front["objs"]]
            )

    return fronts
