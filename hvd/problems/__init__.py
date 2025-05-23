import os

from .base import ConstrainedMOOAnalytical, MOOAnalytical
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
from .misc import CONV3, CONV4, CONV4_2F, UF7, UF8
from .pymoo_wrapper import PymooProblemWithAD, PymooProblemWrapper
from .zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6

# enable double-precision of JAX
os.environ["JAX_ENABLE_X64"] = "True"

__all__ = [
    "MOOAnalytical",
    "ConstrainedMOOAnalytical",
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
    "UF7",
    "UF8",
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
    "PymooProblemWithAD",
    "PymooProblemWrapper",
]
