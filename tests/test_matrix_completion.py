import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

from nopt.transforms import *
from nopt.constraints import *
from nopt.problems import *
from nopt.solvers import *
from nopt.benchmarks import *


if __name__ == "__main__":
    m = 100
    n = 100

    r = 10
    delta = 0.5
    p = round(delta*m*n)

    x0, _ = GenerateLSMat1((m,n), r, 0, 1)
    HTr = FixedRank(r, (m,n))
    mask = np.random.choice(m*n,p,replace = False)
    P_omega = EntryWise(mask, m*n)
    b = P_omega.matvec(x0.flatten())

    problem = LinearProblem(P_omega, b, HTr)
    solver = NIHT(logverbosity = 2, verbosity = 2)
    x, opt_log = solver.solve(problem)