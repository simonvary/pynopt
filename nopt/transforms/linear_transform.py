"""
    2021, Simon Vary
Module providing basic 1D transforms
Routine listings
----------------
get_function_handle(method)
    Return a function handle to a given threshold operator.
threshold_weighted_soft(var)
    The weighted soft threshold operator.
"""

from nopt.transforms.transform import Transform

import numpy as np

class LinearMatrix(Transform):
    """
    Linear transform based on a numpy array
    """
    
    def __init__(self, A, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._A = A
        self.m = A.shape[0]
        self.input = A.shape[1]

    # Function to apply the transform.
    def __call__(self, x):
        return np.matmul(self._A, x)

    # Function to apply adjoint/backward transform.
    def adjoint(self, y):
        return np.matmul(self._A.transpose(), y)