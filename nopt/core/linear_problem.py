"""
Module containing pymanopt problem class. Use this to build a problem
object to feed to one of the solvers.
"""


from nopt.core.problem import Problem
import numpy as np

class LinearProblem(Problem):
    """
    Problem class for setting up a problem to feed to one of the PyLAS solvers.
    Attributes:
        - A
            A linear operator going from R^{m\times n} to R^p
            If none given, an identity is assumed
        - b
            A right-hand side in R^p
        - cost
            A callable which takes an element of R^p and returns a real number.
        - constraint_type
            A string, either:
                'lr':  low-rank
                'lrps': low-rank plus sparse
                'lras': low-rank and sparse
        - verbosity (2)
            Level of information printed by the solver while it operates, 0
            is silent, 2 is most information.
    Methods:
        - cost
            Least squares cost
        - gradient
            Gradient of the cost in x
    """

    def __init__(self, A, b, constraint, x_true=None, verbosity=2):
        
        # Here should be probably class
        if A is None:
            def A(x):
                return x
        
        self.A = A
        self.b = b
        self.constraint = constraint
        self.x_true = x_true
        self.verbosity = verbosity # possibly not needed?
    
    def objective(self, x):
        # Least squares cost
        return .5*np.linalg.norm(self.A(x) - self.b, 2)**2