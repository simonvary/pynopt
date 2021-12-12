import time
import numpy as np
from nopt.constraints.sparse_oblique import SparseOblique
from nopt.constraints import FixedRank, Sparsity

from nopt.solvers.solver import Solver
import pdb

class ObliqueSPCA(Solver):

    def __init__(self, linesearch='normalized', *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def _compute_initial_guess(self, A, problem):
        HTr = FixedRank(problem.rank)
        HTso = SparseOblique(problem.sparsity)
        subspaces,_ = HTr.project(A._matrix)
        _, x0 = HTso.project(subspaces[1])
        return (x0)

    def _take_step(self, x, alpha, direction, projection):
        w_new = x + alpha * direction
        subspace, x_new = projection(w_new)
        return(subspace, x_new)
    
    def solve(self, problem, lam=None, x0=None):
        """
        Perform optimization using gradient descent with linesearch.
        This method first computes the gradient (derivative) of obj
        w.r.t. arg, and then optimizes by moving in the direction of
        steepest descent (which is the opposite direction to the gradient).
        Arguments:
            - problem (PrincipalSubspace)
            - x0=None
                Optional parameter. Starting point. If none
                then a starting point will be randomly generated.
            - stepsize_type = barmijo
                Whether to reuse the previous linesearch object. Allows to
                use information from a previous solve run.
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
        """

        A = problem.A
        verbosity = problem.verbosity
        s = problem.sparsity
        r = problem.rank
        n = np.prod(A.shape_input)

        if lam is None:
            lam = n
        
        if x0 is None:
            x = self._compute_initial_guess(A, problem)
        else:
            x = x0
        
        regularizer = lambda x: .25*lam*np.linalg.norm(x.T @ x - np.eye(x.shape[1]),'fro')**2
        regularizer_gradient = lambda x: lam * x @ (x.T @ x - np.eye(x.shape[1]))
        
        objective = lambda x: problem.objective(x) + regularizer(x)
        gradient = lambda x: problem.gradient(x) + regularizer_gradient(x)

        extraiterfields = ['gradnorm', 'iter_lsearch', 'stepsize']
        if problem.x_true is not None:
            fx_true = problem.objective(problem.x_true)
            extraiterfields.append('dist_fx_true')
            extraiterfields.append('dist_x_true')
        else:
            dist_fx_true = None
            dist_x_true = None
        
        MAX_ITER_LSEARCH = 100;

        alpha_bar = 1
        tau = 0.5
        beta = 1e-4

        HTso = SparseOblique(s)

        objective_value = objective(x)
        if verbosity >= 2:
            print(" iter\t\t   obj. value\t    grad. norm\t     iter. lsearch\t     stepsize")

        self._start_optlog(None, extraiterfields)
        stop_reason = None
        iter = 0
        time0 = time.time()
        while True:
            # Calculate new cost, grad and gradnorm
            grad = gradient(x)
            gradnorm = np.linalg.norm(grad, 2)

            alpha = alpha_bar
            iter_lsearch = 1
            while True:
                subspace, x_new = self._take_step(x, alpha, -grad, HTso.project)
                s_new = x_new - x
                objective_value_new = objective(x_new)
                if objective_value - objective_value_new >= beta*( - np.dot(s_new.flatten(), grad.flatten())) or iter_lsearch > MAX_ITER_LSEARCH:
                    break
                alpha = alpha * tau
                iter_lsearch = iter_lsearch + 1

            x = x_new
            objective_value = objective(x)

            iter = iter + 1
            if verbosity >= 2:
                print("%5d\t%+.16e\t%.8e \t %5d \t\t\t %2.4e" % (iter, objective_value, gradnorm, iter_lsearch, alpha))

            if problem.x_true is not None:
                dist_fx_true = (objective_value - fx_true) / fx_true
                dist_x_true = problem.distance_x_true(x)

            if self._logverbosity >= 2:
                self._append_optlog(iter, time0, objective_value, 
                                    gradnorm = gradnorm, 
                                    iter_lsearch = iter_lsearch,
                                    stepsize = alpha,
                                    dist_fx_true = dist_fx_true, 
                                    dist_x_true = dist_x_true)

            stop_reason = self._check_stopping_criterion(
                time0, iter=iter, objective_value=objective_value, stepsize=alpha, gradnorm=gradnorm)

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

        
        if self._logverbosity <= 0:
            return (subspace, x)
        else:
            self._stop_optlog(stop_reason, iter, time0, objective(x),
                              stepsize=alpha, 
                              gradnorm=gradnorm, 
                              dist_fx_true = dist_fx_true, 
                              dist_x_true = dist_x_true)
            return (subspace, x, self._optlog)