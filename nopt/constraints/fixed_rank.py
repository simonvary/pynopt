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
from sklearn.utils.extmath import randomized_svd


class FixedRank(Constraint):
    """
    Projections based on matrix rank
    """

    def __init__(self, r, matrix_shape, randomized=True):
        self.r = r
        self.randomized = randomized
        self.matrix_shape = matrix_shape
        
    def project(self, x, r=None, factorized = False):
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
            
        randomized = self.randomized
        shape = self.matrix_shape

        if randomized:
            U, S, V = randomized_svd(x.reshape(shape), n_components=r, random_state=None)
        else:
            U, S, V = np.linalg.svd(x.reshape(shape), full_matrices=False)
        V = V.transpose()
        S = np.diag(S[:r])    
        U = U[:,:r]
        V = V[:,:r]
        if factorized:
            S_sqrt = np.sqrt(S)
            return (U, V), [U @ S_sqrt, S_sqrt@V.T]
        else:
            return (U, V), np.matmul(U, np.matmul(S, V.transpose())).flatten()


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
        shape = self.matrix_shape
        return np.matmul(U, np.matmul(U.transpose(), x.reshape(shape))).flatten()

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