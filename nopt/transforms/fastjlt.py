"""
    2021, Simon Vary
    Module providing Fast Johnson-Lindenstrauss transform
    
    Routine listings
    ----------------
    get_function_handle(method)
        Return a function handle to a given threshold operator.
    threshold_weighted_soft(var)
        The weighted soft threshold operator.

"""

from nopt.transforms.transform import Transform

import numpy as np

class FastJLT(Transform):
    """
    Linear transform based on a numpy array
    """
    
    def __init__(self, shape_input, shape_output, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.R = None #sparse_identity with random p rows selected
        self.D = None #sparse diagonal matrix with +-1

    # Function to apply the transform.
    def matvec(self, x):
        
        return np.fft.rfft(x)

    # Function to apply adjoint/backward transform.
    def rmatvec(self, y):
        return np.fft.irfft(y)