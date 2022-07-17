#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue March 1 2022

@author: Simon
"""

import time
import numpy as np

class LMaFit(Solver):
    """r
    LMaFit: Low-Rank Matrix Fitting

    http://lmafit.blogs.rice.edu
    """
    
    def __init__(self, linesearch='normalized', *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    
    def _compute_initial_guess(self, A, b, constraints):
        w = A.rmatvec(b)
        T_1, x1 = constraints[0].project(w)
        T_2, x2 = constraints[1].project(w - x1) 
        return [[T_1, T_2], [x1, x2]]

    def solve(self, problem, x=None):
        """
        LMaFit for the recovery of low-rank matrix
        Arguments:
            - problem
                Nopt problem setup using the Problem class, this must
                have a .manifold attribute specifying the manifold to optimize
                over, as well as a cost and enough information to compute
                the gradient of that cost.
            - x=None
                Optional parameter. Starting point. If none
                then a starting point will be computed from A.rmatvec(b).
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
            - optlog
        """

        # Check the problem is LinearProblemSum type
        constraint = problem.constraint
        objective = problem.objective
        lambda_x = problem.lambda_x
        lambda_x = problem.lambda_y
        eye = np.eye(constraint.r)

        A = problem.A
        b = problem.b
        verbosity = problem.verbosity

        # x is now a tuple x = (x1, x2)

        if x is None:
            subspaces, x = self._compute_initial_guess(A, b, constraints)
        else:
            subspaces, _ = constraints.project(x) # broken now

        if verbosity >= 2:
            print(" iter\t\t   obj. value\t    grad. norm")

        self._start_optlog()
        stop_reason = None
        iter = 0
        time0 = time.time()

        if verbosity >= 2:
            print(" iter\t\t   cost val\t    grad. norm")

        while True:
            # Calculate new cost, grad and gradnorm
            # objective_value = objective(x[0] + x[1])
            iter = iter + 1

            # Least squares for X
            Xt = np.linalg.solve(Y@Y.T + lambda_x*eye, Y@Z.T)
            
            # Least squares for Y
            Y = np.linalg.solve(Xt@X + lambda_y*eye, Xt@Z)

            Z = Xt.T @ Y

            objective_value = objective(Z)

            if verbosity >= 2:
                print("%5d\t%+.16e\t%.8e" % (iter, objective_value, gradnorm))

            if self._logverbosity >= 2:
                self._append_optlog(iter, objective_value, xdist = None) # gradnorm=gradnorm

            stop_reason = self._check_stopping_criterion(
                time0, iter=iter, objective_value=objective_value, stepsize=alpha, gradnorm=gradnorm)

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

        
        if self._logverbosity <= 0:
            return x
        else:
            self._stop_optlog(x[0] + x[1], objective(x[0] + x[1]), stop_reason, time0,
                              stepsize=alpha, gradnorm=gradnorm,
                              iter=iter)
            return x, self._optlog

def lmafit2(mat, mask, rank, maxit=50, kappa = 1-1e-4, tol_res_error = 1e-6):
    '''
        Based on a pseudocode provided in:
            Tanner, Wei. Low rank matrix completion by alternating steepest descent methods. ACHA 2016 
            https://www.sciencedirect.com/science/article/pii/S1063520315001062' 

        mat is a matrix with known entries on the mask
        mask be the boolean matrix with True for known values

        Xt.T @ Y = Z
        Xt is X transposed
    '''

    b = mat[mask] # The vector of observed entries
    
    m, n = mat.shape
    
    # Default conditioning to machine precision times max(m,n)
    tol_rank = -1 # default conditioning to machine precision times max(m,n)

    Z = np.zeros((m,n))
    Z[mask] = b
    Z[~mask] = b.mean()
    
    # Initialization
    _, S, Vh = randomized_svd(Z,rank, random_state = None)
    Y = np.diag(S)  @ Vh

    rel_error = np.zeros((maxit+1,1))
    rel_error_kappa = -1 # only computed in iteration 15
    iter = 0
    while True:
        Xt, _, _, _ = np.linalg.lstsq(Y.T, Z.T, rcond = tol_rank)
        Y, _, _, _ = np.linalg.lstsq(Xt.T, Z, rcond = tol_rank)
        Z = Xt.T @ Y

        rel_error[iter] = np.linalg.norm(Z[mask]-b, 2) / np.linalg.norm(b, 2)

        if iter>=15:
            rel_error_kappa = (rel_error[iter] / rel_error[iter-15])**(1/15)

        if (iter >= maxit) or (rel_error[iter] <= tol_res_error) or (rel_error_kappa > kappa)  : 
            break
        else:
            Z[mask] = b
            iter = iter + 1

    rel_error = rel_error[:iter]

    return Xt.T@Y, rel_error, None
