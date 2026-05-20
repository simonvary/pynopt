"""
    2021, Simon
Module providing thresholding operators for algorithms
Routine listings
----------------
thresh_hard_sparse(method)
    Return a function handle to a given threshold operator.
support_projection(var)
    The sparsity hard thresholding operator.
"""

import numpy as np

from nopt.constraints.constraint import Constraint

class JointConstraint(Constraint):
    '''
    Takes a list of constraints that will be applied in a sequence.
    '''
    def __init__(self, constraints, dtype=None, *args, **kwargs):
        super().__init__(dtype, shape=(As[-1].shape[0], As[0].shape[1]), *args, **kwargs)
        self._constraints = constraints
        self._n_constraints = len(constraints)

    def project(self, x, tol=1e-6, max_it = 100):
        y = x.copy()
        delta = 1
        it = 1
        while delta >= tol and it <= max_it:
            y_old = y.copy()
            subspace, y = self._constraints[0].project(y)
            for constraint in self._constraints[1:]:
                y = constraint.project(y)
            delta = np.linalg.norm(y_old - y) / np.linalg.norm(y_old)
            it = it + 1
        subspace, y = self._constraints[0].project(y)
        return subspace, y

    def project_subspace(self, x, subspace):
        return(self._constraints[0].project_subspace(x, subspace))