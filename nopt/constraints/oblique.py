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

import numpy as np

from nopt.constraints.constraint import Constraint

class Oblique(Constraint):
    """
    Projections based on an oblique manifold
    """

    def __init__(self):
        None

    def project(self, x):
        """
        Normalize the columns of x.
        Parameters
        ----------
        x : numpy 2D array
            Numpy array to be projected
        -----
        """
        col_norms = np.linalg.norm(x, ord=2, axis=0)
        return (x / col_norms)


    def project_subspace(self, x, subspaces):
        """
        Keeps only parameters at specified indices setting others to zero
        ----------
        x : numpy array
            Numpy array to be projected
        ind : int
            where to keep entries
        """
        # no implementation
        return None

    def project_subspace_right(self, x, subspaces):
        """
        Keeps only parameters at specified indices setting others to zero
        ----------
        x : numpy array
            Numpy array to be projected
        ind : int
            where to keep entries
        """
        # missing implementation yet
        return None