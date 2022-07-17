import numpy as np
import matplotlib.pyplot as plt

from nopt.transforms.linear_transform import LinearMatrix
from nopt.constraints.sparsity import *
from nopt.constraints.rank import *
from nopt.problems.linear_problem import *
from nopt.solvers.niht import *



m = 1000
n = 400
Amat = np.random.normal(0,1, (m,n))
A = LinearMatrix(Amat)
k = 200
r = 20
HTs = Sparsity(k)
HTr = Rank(r)

x_true = np.random.normal(0,1, (n, ))
_, x_true = HTs.project(x_true)
b = A.matvec(x_true)


solver = NIHT(logverbosity = 2)
problem = LinearProblem(A, b, HTs)


x, opt_log = solver.solve(problem)


plt.semilogy(opt_log['iterations']['fx'])
