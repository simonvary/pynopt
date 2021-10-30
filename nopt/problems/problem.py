"""
Module containing pymanopt problem class. Use this to build a problem
object to feed to one of the solvers.
"""

import numpy as np

class Problem:
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
        - constraint
            nopt::constraints::constraint
        - verbosity (2)
            Level of information printed by the solver while it operates, 0
            is silent, 2 is most information.
    """

    def __init__(self, x_true=None, verbosity=2):
        self.x_true = x_true
        self.verbosity = verbosity

