"""
    2022, Simon
BlockFixedRank operation
"""


import numpy as np

from nopt.constraints.constraint import Constraint
from sklearn.utils.extmath import randomized_svd


class BlockFixedRank(Constraint):
    """
    Projections based on matrix rank
    """

    def __init__(self, block_regions, ranks, matrix_shape, randomized=True):
        if type(ranks) == int:
            self.ranks = [ranks] * len(block_regions)
        else:
            self.ranks = ranks
        self.block_regions = block_regions
        self.randomized = randomized
        self.matrix_shape = matrix_shape
        
    def project(self, x, block_regions=None, ranks=None, factorized = False):
        """
        
        """
        if ranks is None:
            ranks = self.ranks
        if type(ranks) == int:
            ranks = [ranks] * len(block_regions)
        if block_regions is None:
            block_regions = self.block_regions

            
        randomized = self.randomized
        shape = self.matrix_shape

        x_matrix = x.reshape(shape)
        y_matrix = np.zeros_like(x_matrix)
        sing_vectors = []

        for r, region in zip(ranks, block_regions):
            matrix_region = x_matrix[:,region]
            if randomized:
                U, S, V = randomized_svd(matrix_region, n_components=r, random_state=None)
            else:
                U, S, V = np.linalg.svd(matrix_region, full_matrices=False)
            V = V.transpose()
            S = np.diag(S[:r])    
            U = U[:,:r]
            V = V[:,:r]
            y_matrix[:,region] = np.matmul(U, np.matmul(S, V.transpose()))
            sing_vectors.append((U, V))
        return (block_regions, sing_vectors), y_matrix.flatten()


    def project_subspace(self, x, subspaces):
        """
        Keeps only parameters at specified indices setting others to zero
        ----------
        x : numpy array
            Numpy array to be projected
        ind : int
            where to keep entries
        """
        x_matrix = x.reshape(self.matrix_shape)
        y_matrix = np.zeros_like(x_matrix)

        for region, (U,V) in zip(*subspaces):
            y_matrix[:,region] = np.matmul(U, np.matmul(U.transpose(), x_matrix[:,region]))
        return y_matrix.flatten()

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