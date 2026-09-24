"""Maximum mean discrepancy indicators.

Kernels live in ``hvd.mmd.kernels``; the Newton optimizer lives in
``hvd.mmd_newton``. The original indicators remain in ``hvd.mmd.legacy``
for comparison with the vectorized implementation.
"""

from .vectorized import MMD, MMDMatching

__all__ = ["MMD", "MMDMatching"]
