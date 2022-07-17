"""
    2021, Simon Vary
Module providing basic 1D transforms
Routine listings
----------------

"""

import numpy as np

from nopt.transforms.transform import Transform
from scipy.sparse.linalg import LinearOperator 
from scipy.sparse import coo_matrix

class EntryWise(Transform):
    """
    Entry-wise transform on a sparse mask

    # should take the scipy LinearOperator for inheritence
    """  
    def __init__(self, mask, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._mask = mask
        # or dia = mask.reshape(-1).astype('float').tolist()
        # subsample2 = sparse.diags(dia)

        if self.shape_input == None:
            self.shape_input = mask.shape
        if self.shape_output == None:
            self.shape_output= mask.sum()

    # Function to apply the transform.
    def matvec(self, x):
        # change x from the self.shape_input to a vector
        #
        return x[self._mask].reshape(-1,1)

    # Function to apply adjoint/backward transform.
    def rmatvec(self, y):
        '''
            fill in the entries of a zero matrix of a shape
        '''
        return np.matmul(self._matrix.transpose(), y).reshape(self.shape_input)

    # Function to apply the transform.
    def matmat(self, x):
        # Todo: apply matvec in vmap
        return np.matmul(self._matrix, x)

    # Function to apply adjoint/backward transform.
    def rmatmat(self, y):
        # change result from a vector to the self.shape_input 
        # check that y is a matrix
        return np.matmul(self._matrix.transpose(), y)