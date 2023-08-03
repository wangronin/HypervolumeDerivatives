import numpy as np

from hvd.hybrid import NSGA_DpN
from hvd.problems import Eq1DTLZ1, Eq1DTLZ2, Eq1DTLZ3, Eq1DTLZ4, Eq1IDTLZ1, Eq1IDTLZ2, Eq1IDTLZ3, Eq1IDTLZ4

N = 15
problems = [
    Eq1DTLZ1(3, 11),
    Eq1DTLZ2(3, 11),
    Eq1DTLZ3(3, 11),
    Eq1DTLZ4(3, 11),
    Eq1IDTLZ1(3, 11),
    Eq1IDTLZ2(3, 11),
    Eq1IDTLZ4(3, 11),
]

for problem in problems:
    algorithm = NSGA_DpN(problem, 500, 10)
    out = algorithm.run()
    # data = Parallel(n_jobs=N)(delayed(hybrid)(i, problem, ref) for i in range(N))
    # np.savez(f"{type(problem).__name__}-hybrid.npz", data=data)
