import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

from nopt.transforms import *
from nopt.constraints import *
from nopt.problems import *
from nopt.solvers import *
from nopt.benchmarks import *
import scipy

if __name__ == "__main__":
    n = 1000
    delta = 0.3
    rho = 0.3
    p = round(n*delta)
    Amat = np.random.normal(0,1, (p,n))  / np.sqrt(n)
    A = scipy.sparse.linalg.aslinearoperator(Amat)

    s = round(p * rho)
    HTs = Sparsity(s)

    x0 = np.random.normal(0,1, (n, 1)) / np.sqrt(n)
    sub, x0 = HTs.project(x0)
    b = A.matvec(x0)

    problem = LinearProblem(A, b, HTs, x_true = x0)

    solver = NIHT(logverbosity = 2, verbosity=2)
    x, opt_log = solver.solve(problem)