"""
Module containing pymanopt problem class. Use this to build a problem
object to feed to one of the solvers.
"""

import numpy as np

from nopt.problems.problem import Problem

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
    def __init__(self, A, rank, sparsity, lam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = A
        self.sparsity = sparsity
        self.rank = rank
        self.lam = lam # this lam should be part of the solver really
    
    def objective(self, x):
        # Least squares cost where x is an n times r matrix 
        return ( .5*np.linalg.norm(self.A._matrix, 'fro')**2 -.5*np.linalg.norm(self.A.matmat(x), 'fro')**2  
                    + self.lam/4 * np.linalg.norm(x.T @ x - np.eye(x.shape[1]),'fro')**2)

    def gradient(self, x):
        return ( -(self.A._matrix.T @ self.A._matrix) @ x  
                    + self.lam * x @ (x.T @ x - np.eye(x.shape[1])) )