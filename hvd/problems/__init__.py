import os

from .base import CMOP, MOP, JaxFunction
from .cf import CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10
from .dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
from .eqdtlz import (
    IDTLZ1,
    IDTLZ2,
    IDTLZ3,
    IDTLZ4,
    Eq1DTLZ1,
    Eq1DTLZ2,
    Eq1DTLZ3,
    Eq1DTLZ4,
    Eq1IDTLZ1,
    Eq1IDTLZ2,
    Eq1IDTLZ3,
    Eq1IDTLZ4,
)
from .factory import get_pymoo_problem, pymoo_problem_names
from .misc import CONV3, CONV4, CONV4_2F
from .uf import UF1, UF2, UF3, UF4, UF5, UF6, UF7, UF8, UF9, UF10
from .wfg import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
from .zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6

# enable double-precision of JAX
os.environ["JAX_ENABLE_X64"] = "True"

__all__ = [
    "MOP",
    "CMOP",
    "JaxFunction",
    "get_pymoo_problem",
    "pymoo_problem_names",
    "Eq1DTLZ1",
    "Eq1DTLZ2",
    "Eq1DTLZ3",
    "Eq1DTLZ4",
    "Eq1IDTLZ1",
    "Eq1IDTLZ2",
    "Eq1IDTLZ3",
    "Eq1IDTLZ4",
    "IDTLZ1",
    "IDTLZ2",
    "IDTLZ3",
    "IDTLZ4",
    "CF1",
    "CF2",
    "CF3",
    "CF4",
    "CF5",
    "CF6",
    "CF7",
    "CF8",
    "CF9",
    "CF10",
    "CONV3",
    "CONV4",
    "CONV4_2F",
    "UF1",
    "UF2",
    "UF3",
    "UF4",
    "UF5",
    "UF6",
    "UF7",
    "UF8",
    "UF9",
    "UF10",
    "WFG1",
    "WFG2",
    "WFG3",
    "WFG4",
    "WFG5",
    "WFG6",
    "WFG7",
    "WFG8",
    "WFG9",
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "ZDT4",
    "ZDT6",
    "DTLZ1",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ6",
    "DTLZ7",
]
