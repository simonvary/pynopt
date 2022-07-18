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
import scipy.sparse as sp
import scipy.fftpack as fftpack
from scipy.sparse.linalg import LinearOperator


class FastJLT(LinearOperator):
    """
    Linear transform based on a numpy array
    """

    def _generate(self, shape):
        R_ind = np.sort(np.random.choice(shape[1], size=shape[0], replace=False))
        D = sp.diags(np.random.choice((1, -1), size = shape[1]))
        return (R_ind, D)

    def __init__(self, shape, dtype = None, *args, **kwargs):
        super().__init__(dtype, shape, *args, **kwargs)
        self.R_ind, self.D = self._generate(self.shape)

    # Function to apply the transform.
    def _matvec(self, x):
        w = self.D.dot(fftpack.dct(x, norm='ortho'))
        return w[self.R_ind]

    # Function to apply adjoint/backward transform.
    def _rmatvec(self, y):
        w = np.zeros(self.shape[1], dtype = self.dtype)
        w[self.R_ind] = y
        return fftpack.idct(self.D.dot(w), norm='ortho')#.reshape(self.shape_input)
        # still have to add the reshapes
