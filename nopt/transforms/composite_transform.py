import numpy as np

from nopt.transforms.transform import Transform
from scipy.sparse.linalg import LinearOperator


class CompositeTransform(LinearOperator):
    '''
    Takes a list of transforms that will be applied in a sequence.
    '''
    def __init__(self, As, dtype=None, *args, **kwargs):
        super().__init__(dtype, shape=(As[-1].shape[0], As[0].shape[1]), *args, **kwargs)
        self._As = As
        self._n_transforms = len(As)

    def _matvec(self, x):
        y = x.copy()
        for A in self._As:
            y = A.matvec(y)
        return y

    def _rmatvec(self, y):
        x = y.copy()
        for A in reversed(self._As):
            x = A.rmatvec(x)
        return x
