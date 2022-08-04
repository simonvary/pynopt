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
            Can be a list of the same length as constraints. Nones will be turned into identity mappings.
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

    def __init__(self, A, b, constraints,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._num_components = len(constraints)

        if type(A) is tuple:
            self.A = []
            for i in range(self._num_components):
                if A[i] is None:
                    class Ai(object):
                        def __init__(self):
                            pass
                        def matvec(self, x):
                            return x
                        def rmatvec(self, y):
                            return y
                else:
                    Ai = A[i]
                self.A.append(Ai)
            self.A = tuple(self.A)
        else:
            if A is None:
                class Ai(object):
                    def __init__(self):
                        pass
                    def matvec(self, x):
                        return x
                    def rmatvec(self, y):
                        return y
            self.A = (A,) * self._num_components
        self.b = b
        # check that there are at least two constraints
        self.constraints = constraints
    
    def objective(self, x):
        # Least squares cost
        Ax = np.zeros_like(self.b)
        for i in range(self._num_components):
            Ax += self.A[i].matvec(x[i])
        return .5*np.linalg.norm(Ax - self.b, 2)**2