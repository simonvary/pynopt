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

class Sparsity(Constraint):
    """
    Projections of tensors based on sparsity
    """

    def __init__(self, k):
        self.k = k

    def project(self, x, k=None):
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
        if k is None:
            k = self.k
        _x = x.copy()

        if isinstance(k, int):
            ind = np.argpartition(np.abs(x), -k, axis=None)[-k:]
            ind = np.unravel_index(ind, _x.shape)
            ind_del = np.ones_like(_x, dtype=bool)
            ind_del[ind] = False
            _x[ind_del] = 0
        else:
            ind_tmp = np.argpartition(np.abs(_x), kth=-k, axis=0)
            ind_keep = np.ones_like(_x, dtype=bool)
            for i in range(x.shape[1]):
                ind_keep[ind_tmp[:-k[i],i],i] = False
            _x[np.logical_not(ind_keep)] = 0
            ind = np.where(ind_keep)
        return np.sort(ind[0]), _x

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
        return _x