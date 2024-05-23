import numpy as np

from nopt.transforms.transform import Transform
from scipy.sparse.linalg import LinearOperator




class RasterScan(LinearOperator):
    """
    RasterScan transform on a sparse mask
    """  

    def _generate_mask(self, shape, rows_per_band):
        n_1, n_2, n_E = shape
        mask = np.zeros(shape, dtype=bool)
        for i in range(n_E):
            ind = np.random.choice(n_1, rows_per_band, replace = False)
            mask[ind,:,i] = True
        mask = np.where(mask.flatten())
        return mask
    
    def __init__(self, input_shape, rows_per_band , dtype = None, *args, **kwargs):
        # check that shape[0] == len(mask)
        super().__init__(dtype, (input_shape[1]*input_shape[2]*rows_per_band, np.prod(input_shape)), *args, **kwargs)
        self._mask = self._generate_mask(input_shape, rows_per_band)

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