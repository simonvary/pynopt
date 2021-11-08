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


class FastJLT(Transform):
    """
    Linear transform based on a numpy array
    """

    def _generate(self, n, p):
        R_ind = np.sort(np.random.choice(n, size=p, replace=False))
        D = sp.diags(np.random.choice((1, -1), size = n))
        return (R_ind, D)

    def __init__(self, shape_input, shape_output, *args, **kwargs):
        super().__init__(shape_input, shape_output, *args, **kwargs)
        self.n = np.prod(shape_input)
        self.p = np.prod(shape_output)
        self.R_ind, self.D = self._generate(self.n, self.p)

    # Function to apply the transform.
    def matvec(self, x):
        w = self.D.dot(fftpack.dct(x.flatten(), norm='ortho'))
        return w[self.R_ind]

    # Function to apply adjoint/backward transform.
    def rmatvec(self, y):
        w = np.zeros(np.prod(self.shape_input))
        w[self.R_ind] = y
        return fftpack.idct(self.D.dot(w), norm='ortho').reshape(self.shape_input)
        # still have to add the reshapes