import numpy as np

from nopt.transforms.transform import Transform
from scipy.sparse.linalg import LinearOperator


class HadamardProduct(LinearOperator):
    """
    Entry-wise transform on a sparse mask
    """  
    def __init__(self, vector, dtype = None, *args, **kwargs):
        # check that shape[0] == len(mask)
        super().__init__(dtype, (np.prod(vector.shape), np.prod(vector.shape)), *args, **kwargs)
        self._vector = vector
        self._vector_inv = 1/vector

    def _matvec(self, x):
        return x * self._vector

    def _rmatvec(self, y):
        '''
            fill in the entries of a zero matrix of a shape
        '''
        return y * self._vector_inv