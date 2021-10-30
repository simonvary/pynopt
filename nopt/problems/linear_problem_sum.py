"""
Module containing pynopt problem class. Use this to build a problem
object to feed to one of the solvers.
"""

import numpy as np

from nopt.problems.problem import Problem

class LinearProblemSum(Problem):
    """
    Problem class for setting up a problem to feed to one of the nopt solvers.
    Attributes:
        - A
            A linear operator going from R^{m\times n} to R^p
            If none given, an identity is assumed
        - b
            A right-hand side in R^p
        - cost
            A callable which takes an element of R^p and returns a real number.
        - constraints list of (nopt::constraint)
        - verbosity (2)
            Level of information printed by the solver while it operates, 0
            is silent, 2 is most information.
    Methods:
        - cost
            Least squares cost
        - gradient
            Gradient of the cost in x
    """

    def __init__(self, A, b, constraints, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if A is None:
            def A(x):
                return x        
        self.A = A
        self.b = b
        # check that there are at least two constraints
        self.constraints = constraints
    
    def objective(self, x):
        # Least squares cost
        return .5*np.linalg.norm(self.A.matvec(x) - self.b, 2)**2