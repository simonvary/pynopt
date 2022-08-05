"""
..
    2022, Simon Vary
Module providing basic 2D transforms
A.forward is supposed to go from 2D matrix domain to a vector of coeffs
A.backward goes from coeffs to a matrix
Routine listings
----------------
wavelet2(method)
    2D wavelet transform.
"""

class CompositeTransform(object):
    '''
    Takes a list of transforms that will be applied in sequence.
    '''
    def __init__(self, As, **kwargs):
        self._As = As

    def matvec(self, x):
        y = x.copy()
        for A in self._As:
            y = A.matvec(y)
        return y

    def rmatvec(self, y):
        x = y.copy()
        for A in reversed(self._As):
            x = A.rmatvec(x)
        return x
