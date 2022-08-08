import numpy as np

from nopt.transforms.transform import Transform
from scipy.sparse.linalg import LinearOperator


class EntryWise(LinearOperator):
    """
    Entry-wise transform on a sparse mask
    """  
    def __init__(self, shape, mask, dtype = None, *args, **kwargs):
        # check that shape[0] == len(mask)
        super().__init__(dtype, shape, *args, **kwargs)
        self._mask = mask

    def _matvec(self, x):
        return x[self._mask]

    def _rmatvec(self, y):
        '''
            fill in the entries of a zero matrix of a shape
        '''
        _x = np.zeros(self.shape[1], dtype = self.dtype)
        _x[self._mask] = y
        return _x

    def project(self, y):
        '''
            fill in the entries of a zero matrix of a shape
        '''
        _x = np.zeros(self.shape[1], dtype = self.dtype)
        _x[self._mask] = y
        return _x