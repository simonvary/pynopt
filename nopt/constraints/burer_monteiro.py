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

class FixedRank(Constraint):
    """
    Projections based on matrix rank
    """

    def __init__(self, r):
        self.r = r

    def project(self, x, r=None):
        """
        Keep only k largest entries of x.
        Parameters
        ----------
        x : numpy 2D array
            Numpy array to be thresholded
        r : int
            Rank
        Notes
        -----
        This is hard thresholding, keeping k largest entries in absolute value
        """
        if r is None:
            r = self.r
        U, S, V = np.linalg.svd(x)
        V = V.transpose()
        S = np.diag(S[:r])    
        U = U[:,:r]
        V = V[:,:r]
        return (U, V), np.matmul(U, np.matmul(S, V.transpose()))


    def project_subspace(self, x, subspaces):
        """
        Keeps only parameters at specified indices setting others to zero
        ----------
        x : numpy array
            Numpy array to be projected
        ind : int
            where to keep entries
        """
        U = subspaces[0]
        V = subspaces[1]
        return np.matmul(U, np.matmul(U.transpose(), x))

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