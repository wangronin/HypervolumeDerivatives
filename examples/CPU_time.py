import time

import numpy as np
from hvd.problems import Eq1DTLZ1

f = Eq1DTLZ1()

t0 = time.process_time_ns()
for i in range(100):
    X = np.random.rand(200, 11)
    Y = [f.objective(x) for x in X]
    h = [f.constraint(x) for x in X]
t1 = time.process_time_ns()

print((t1 - t0) / 100 / 1e9)
