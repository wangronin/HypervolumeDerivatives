import numpy as np
from autograd import hessian, jacobian
from optproblems.wfg import WFG1

f = WFG1(3, 5, 2)

x = np.random.rand(5)
print(f.objective_function(x))
jac = jacobian(f.objective_function)
print(jac(x))
