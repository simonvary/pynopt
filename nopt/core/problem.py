"""
Module containing pymanopt problem class. Use this to build a problem
object to feed to one of the solvers.
"""
import functools

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
        - constraint_type
            A string, either:
                'lr':  low-rank
                'lrps': low-rank plus sparse
                'lras': low-rank and sparse
        - verbosity (2)
            Level of information printed by the solver while it operates, 0
            is silent, 2 is most information.
    """

    def __init__(self, A, b, constraint, cost, init, precon=None, verbosity=2):
        
        # Here should be probably class
        pass
