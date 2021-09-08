"""
..
    2021, Simon Vary
Module providing basic 2D transforms
A.forward is supposed to go from 2D matrix domain to a vector of coeffs
A.backward goes from coeffs to a matrix
Routine listings
----------------
wavelet2(method)
    2D wavelet transform.
"""

import numpy as np
import pywt

class Wavelet2(object):
    def __init__(self, wavelet_name, **kwargs):
        self._wavelet_name = wavelet_name
        self._mode = kwargs.get('mode', 'symmetric')
        self._level = kwargs.get('level', None)
        self._coeff_slices = None
        self._coeff_shapes = None

    def forward(self, x):
        coeffs = pywt.wavedecn(x,
                            self._wavelet_name,
                            mode = self._mode,
                            level = self._level)
        # First forward pass initializes the transform
        # by introducing dimensions and wavelet shapes.
        # Or if the size of an input changed from the previous input
        if (self._coeff_slices==None) or (self._coeff_shapes == None) or (self.n != x.size):
            coeff_vec, self._coeff_slices, self._coeff_shapes = pywt.ravel_coeffs(coeffs)
            self.m = coeff_vec.size
            self.n = x.size
        else:
            coeff_vec, _, _ = pywt.ravel_coeffs(coeffs)
        return coeff_vec

    def backward(self, y, **kwargs):
        if (self._coeff_slices==None) or (self._coeff_shapes == None):
            print('Warning! Cannot do backward() before initialized forward()!')
            return(False)
        else:
            out = pywt.unravel_coeffs(y, self._coeff_slices, self._coeff_shapes)
            out = pywt.waverecn(out, self._wavelet_name, mode = self._mode)
            return out