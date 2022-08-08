"""
    2021, Simon
Module providing thresholding operators for algorithms
Routine listings
----------------
thresh_hard_sparse(method)
    Return a function handle to a given threshold operator.
support_projection(var)
    The sparsity hard thresholding operator.
"""

from nopt.constraints.constraint import Constraint

class Euclidean(Constraint):
    """
    No constraint
    """

    def __init__(self):
        pass
        
    def project(self, x):
        """
        Do nothing
        """ 
        return None, x


    def project_subspace(self, x, subspaces):
        """
        Do nothing
        """ 
        return x
