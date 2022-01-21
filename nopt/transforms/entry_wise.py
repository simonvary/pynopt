"""
    2021, Simon Vary
Module providing basic 1D transforms
Routine listings
----------------

"""

import numpy as np

from nopt.transforms.transform import Transform
from scipy.sparse.linalg import LinearOperator 


class EntryWise(Transform):
    """
    Entry-wise transform on a sparse mask

    # should take the scipy LinearOperator for inheritence
    """  
    def __init__(self, ind, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ind = ind

        if self.shape_input == None:
            self.shape_input = (matrix.shape[0], 1)
        if self.shape_output == None:
            self.shape_output= (matrix.shape[1], 1)

    # Function to apply the transform.
    def matvec(self, x):
        # change x from the self.shape_input to a vector
        _x = x.copy()
        ind_del = np.ones(x.shape, dtype=bool)
        ind_del[self._ind] = False
        _x[ind_del] = 0
        return _x

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