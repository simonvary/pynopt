"""
..
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

class SparseOblique(Constraint):
    """
    Projections based on sparsity
    """

    def __init__(self, sparsity):
        ''' k can be a positive int or a list'''
        self.sparsity = sparsity

    def project(self, x, sparsity=None):
        """
        Keep only k largest entries of x.
        Parameters
        ----------
        x : numpy array
            Numpy array to be thresholded
        k : int
            Number of largest entries in absolute value to keep
        Notes
        -----
        This is hard thresholding, keeping k largest entries in absolute value
        """
        return self.project_quasi(x, sparsity=sparsity)

    def project_quasi(self, x, sparsity=None):
        """
        Keeps only parameters at specified indices setting others to zero
        ----------
        x : numpy array
            Numpy array to be projected
        ind : int
            where to keep entries
        """
        if sparsity is None:
            sparsity = self.sparsity
        _x = x.copy()
        ind = np.argpartition(np.abs(x), -sparsity, axis=None)[-sparsity:]
        ind = np.unravel_index(ind, _x.shape)
        ind_del = np.ones(_x.shape, dtype=bool)
        ind_del[ind] = False
        _x[ind_del] = 0
        col_norms = np.linalg.norm(_x, ord=2, axis=0)
        return (ind, _x / col_norms)

    def project_subspace(self, x, ind):
        """
        Keeps only parameters at specified indices setting others to zero
        ----------
        x : numpy array
            Numpy array to be projected
        ind : int
            where to keep entries
        """
        # Check if support is of size k ? 
        _x = x.copy()
        ind_del = np.ones(x.shape, dtype=bool)
        ind_del[ind] = False
        _x[ind_del] = 0
        col_norms = np.linalg.norm(_x, ord=2, axis=0)
        return (_x / col_norms)