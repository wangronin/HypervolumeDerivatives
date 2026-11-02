import os

# Configure both first-import and already-imported JAX sessions before any
# local imports, so every hvd entry point uses the package's float64 contract.
os.environ["JAX_ENABLE_X64"] = "True"

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .delta_p import GenerationalDistance, InvertedGenerationalDistance
from .hypervolume_derivatives import HypervolumeDerivatives
from .newton import HVN, DpN

__all__ = ["HypervolumeDerivatives", "HVN", "DpN", "GenerationalDistance", "InvertedGenerationalDistance"]
