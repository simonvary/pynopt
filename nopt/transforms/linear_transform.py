"""
    2021, Simon Vary
Module providing basic 1D transforms
Routine listings
----------------

"""

from nopt.transforms.transform import Transform

from scipy.sparse.linalg import LinearOperator 

import numpy as np

class LinearMatrix(Transform):
    """
    Linear transform based on a numpy array
    """
    
    def __init__(self, matrix, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._matrix = matrix

        if self.shape_input == None:
            self.shape_input = (matrix.shape[0], 1)
        if self.shape_output == None:
            self.shape_output= (matrix.shape[1], 1)

    # Function to apply the transform.
    def matvec(self, x):
        # change x from the self.shape_input to a vector

        return np.matmul(self._matrix, x.flatten())

    # Function to apply adjoint/backward transform.
    def rmatvec(self, y):
        # change result from a vector to the self.shape_input 
        return np.matmul(self._matrix.transpose(), y).reshape(self.shape_input)