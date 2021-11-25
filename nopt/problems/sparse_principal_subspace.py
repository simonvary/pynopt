"""
Module containing pymanopt problem class. Use this to build a problem
object to feed to one of the solvers.
"""

import numpy as np

from nopt.problems.problem import Problem
from nopt.constraints import Sparsity

class SparsePrincipalSubspace(Problem):
    """
    Problem class for setting up a problem to feed to one of the solvers.
    Attributes:
        - A
            A linear operator going from R^n to R^m
        - verbosity (2)
            Level of information printed by the solver while it operates, 0
            is silent, 2 is most information.
    Methods:
        - cost
            Least squares cost
        - gradient
            Gradient of the cost in x
    """
    def __init__(self, A, rank, sparsity, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = A
        self.sparsity = sparsity
        self.rank = rank
        self._dist_threshold = Sparsity(np.ones(rank, dtype = int))
    
    def objective(self, x):
        return ( .5*np.linalg.norm(self.A._matrix, 'fro')**2 -.5*np.linalg.norm(self.A.matmat(x), 'fro')**2 )

    def gradient(self, x):
        return ( ( -self.A._matrix.T @ self.A._matrix) @ x )

    def distance_x_true(self, x, x_true = None):
        if x_true is None:
             x_true = self.x_true
        # Compute a permutation that will align x with x_true
        _, largest_vals = self._dist_threshold.project(x.T @ x_true)
        permutation = np.sign(largest_vals)
        # Compute a Gram-matrix between x_true and permutated x
        inner_prods = permutation.T @ x.T @ x_true - np.eye(self.rank)
        # Largest deviation between the inner products
        np.abs(inner_prods ).max()
        # Note: Remains to be normalized if used on non-unit length vectors
        return  np.abs(inner_prods ).max()