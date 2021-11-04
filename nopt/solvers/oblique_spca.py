import time
import numpy as np
from nopt.constraints.sparse_oblique import SparseOblique
from nopt.constraints.rank import Rank
from nopt.constraints.sparse import Sparse

from nopt.solvers.solver import Solver
import pdb

class ObliqueSPCA(Solver):

    def __init__(self, linesearch='normalized', *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def _compute_stepsize(self, gradient, subspace, A, constraint):
        gradient_proj = constraint.project_subspace(gradient, subspace)
        a = np.linalg.norm(gradient_proj)**2
        b = np.linalg.norm(A.matvec(gradient_proj))**2
        return a/b

    def _compute_initial_guess(self, A, constraint, problem):
        HTr = Rank(problem.rank)
        subspaces,_ = HTr.project(A._matrix)
        _, x0 = constraint.project_quasi(subspaces[1])
        return (x0)

    def _take_step(self, x, alpha, direction, HTs, HTo):
        w_new = x - alpha * direction
        _, w_new_s = HTs.project(w_new)
        x_new = HTo.project(w_new_s)
        return(x_new)
    
    def solve(self, problem, x=None):
        """
        Perform optimization using gradient descent with linesearch.
        This method first computes the gradient (derivative) of obj
        w.r.t. arg, and then optimizes by moving in the direction of
        steepest descent (which is the opposite direction to the gradient).
        Arguments:
            - problem (PrincipalSubspace)
            - x=None
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

        # Check the problem is LinearLeastSquares type
        objective = problem.objective
        A = problem.A
        verbosity = problem.verbosity
        s = problem.sparsity
        lam = problem.lam

        MAX_ITER_LSEARCH = 100;

        alpha_bar = 1
        tau = 0.5
        beta = 1e-4

        HTso = SparseOblique(s)
        #if x is None:
            #subspace, x = self._compute_initial_guess(A, HTso)
        #else:
            #subspace, _ = constraint.project(x)

        objective_value = objective(x)
        if verbosity >= 2:
            print(" iter\t\t   obj. value\t    grad. norm")

        self._start_optlog()
        stop_reason = None
        iter = 0
        time0 = time.time()
        #pdb.set_trace()
        while True:
            # Calculate new cost, grad and gradnorm
            grad = problem.gradient(x)
            gradnorm = np.linalg.norm(grad, 2)

            alpha = alpha_bar
            # line-search loop
            iter_lsearch = 1
            while True:
                w_new = x - alpha * grad
                #_, w_new_s = HTs.project(w_new)
                #x_new = HTo.project(w_new_s)
                subspace, x_new = HTso.project_quasi(w_new)
                
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
            return (subspace, x)
        else:
            self._stop_optlog(x, objective(x), stop_reason, time0,
                              stepsize=alpha, gradnorm=gradnorm,
                              iter=iter)
            return (subspace, x, self._optlog)