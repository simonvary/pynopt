#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue March 1 2022

@author: Simon
"""

import time
import numpy as np
from nopt.solvers.solver import Solver

class LMaFit(Solver):
    """r
        Based on a pseudocode provided in:
            Tanner, Wei. Low rank matrix completion by alternating steepest descent methods. ACHA 2016 
            https://www.sciencedirect.com/science/article/pii/S1063520315001062' 
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    
    def _compute_initial_guess(self, A, b, constraint):
        w = A.rmatvec(b)
        T_k, x = constraint.project(w, factorized = True)
        return (T_k, x)

    def solve(self, problem, x=None):
        """
        LMaFit for the recovery of low-rank matrix
        Arguments:
            - problem
                Nopt problem setup LinearProblem with FixedRank constraint
            - x=None
                Optional parameter. Starting point. If none
                then a starting point will be computed from A.rmatvec(b).
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
            - optlog
        """

        # Check the problem is LinearProblem with FixedRank constraint
        constraint = problem.constraint
        objective = problem.objective
        #lambda_x = problem.lambda_x
        #lambda_x = problem.lambda_y
        #eye = np.eye(constraint.r)

        A = problem.A
        b = problem.b
        verbosity = self._verbosity


        # x is now a tuple x = (x, yh) and low-rank = x@yh
        if x is None:
            subspaces, x = self._compute_initial_guess(A, b, constraint)
        else:
            subspaces, _ = constraint.project(x)

        if verbosity >= 1:
            print(" Starting")
        if verbosity >= 2:
            print(" iter\t\t   obj. value\t    grad. norm")

        self._start_optlog()
        stop_reason = None
        iter = 0
        time0 = time.time()
        z = x[0] @ x[1]
        while True:
            # Calculate new cost, grad and gradnorm
            iter = iter + 1

            z.flatten()[A._mask] = b

            # Least squares for X
            x[0] = np.linalg.solve(x[1]@x[1].T, x[1]@z.T).T
            
            # Least squares for Y
            x[1] = np.linalg.solve(x[0].T@x[0], x[0].T@z)

            z = x[0] @ x[1]

            objective_value = objective(z.flatten())
            running_time = time.time() - time0

            if verbosity >= 2:
                print("%5d\t%+.16e\t%.8e" % (iter, objective_value, 0))

            if self._logverbosity >= 2:
                self._append_optlog(iter, running_time, objective_value)

            stop_reason = self._check_stopping_criterion(
                running_time, iter=iter, objective_value=objective_value)

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

        
        if self._logverbosity <= 0:
            return z
        else:
            self._stop_optlog(iter, time0, 
                    objective((x[0] @ x[1]).flatten()), 
                    stop_reason, 
                    iter=iter)
            return z, self._optlog
