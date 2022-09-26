

import numpy as np

from nopt.constraints import FixedRank, Sparsity
from nopt.problems import LinearProblemSum
from nopt.solvers import NAHT

import vip_hci as vip
import pylops

class Trajectorlet(pylops.LinearOperator):
    def __init__(self, shape, kernel, angles, dtype=None, *args, **kwargs):
        '''
            Params:
                - shape (the dimensions of the data cube)
                - kernel (the psf kernel for convolution)
                - angles (list of rotated angles)
        '''
        self._shape_cube = shape
        self.shape = (np.prod(self._shape_cube), np.prod(self._shape_cube[-2:]))
        self.dtype = dtype
        super().__init__(dtype, self.shape, *args, **kwargs)
        self._kernel = kernel
        self._angles = angles
        self._conv2d = pylops.signalprocessing.Convolve2D(
            N = np.prod(self._shape_cube[-2:]),
            dims=self._shape_cube[-2:],
            h=self._kernel,
            offset = (self._kernel.shape[0]//2+1, self._kernel.shape[1]//2+1)
        )

    def _matvec(self, x): # Forward operation \Psi(c)
       y = self._conv2d.matvec(x)
       cube_out = np.tile(y, (self._shape_cube[0], 1))
       cube_der = vip.preproc.cube_derotate(cube_out.reshape(self._shape_cube), -self._angles, imlib='opencv')
       return cube_der.flatten()

    def _rmatvec(self, y): # Adjoint operation \Psi^T(y)
        der_cube = vip.preproc.cube_derotate(y.reshape(self._shape_cube), self._angles, imlib='opencv')
        x = self._conv2d.rmatvec(der_cube.sum(axis=0).flatten())
        return x


def exoplanet_lrpt(cube, angle_list, r=20, s=10, prad=1, MAX_ITER=30, fwhm=4, asize= 4, psfn = None, normalize = True):
    '''Exoplanet low-rank plus trajectory'''

    m = cube.shape[0]
    n = np.prod(cube.shape[-2:])
    
    # St dev. normalize
    if normalize:
        stdev_frame = cube.std(axis=0)
        cube_in = cube / stdev_frame
    else:
        cube_in = cube
    
    # Prepare constraints
    HTr = FixedRank(r, (m,n))
    HTs = Sparsity(s) # Euclidean()# 
    constraints = (HTr, HTs)

    # Prepare transforms
    kernel = psfn / np.linalg.norm(psfn)
    trajectorlet = Trajectorlet(cube_in.shape, kernel/m**(1/2), angle_list)
    As = (None, trajectorlet)

    b = cube_in.flatten()

    # Define the problem and solver
    problem = LinearProblemSum(As, b, (HTr, HTs))
    solver = NAHT(logverbosity = 2, maxiter = MAX_ITER)

    # Solve
    x, opt_log = solver.solve(problem)

    cube_planet = As[1]._rmatvec(x[1]).reshape(cube_in.shape)
    cube_background = x[0].reshape(cube_in.shape)
    if normalize:
        cube_planet = cube_planet * stdev_frame
        cube_background = cube_background * stdev_frame

    # Flux estimation?
    
    return (cube_planet, cube_background)