import time
from math import inf, sqrt
import numpy as np

#from pymanopt.manifolds import Stiefel

from nopt.constraints.sparse_oblique import SparseOblique
from nopt.constraints import FixedRank, Sparsity

from nopt.solvers.solver import Solver
import pdb

class LandingField(Solver):

    def __init__(self, stepsize_type='safeguard', *args, **kwargs):
        super().__init__(minobjective_value=-inf, *args, **kwargs)
        self.stepsize_type = stepsize_type
        pass

    def _compute_initial_guess(self, A, constraint, problem):
        HTr = FixedRank(problem.rank)
        subspaces,_ = HTr.project(A._matrix)
        _, x0 = constraint.project_quasi(subspaces[1])
        return (x0)

    def _safe_stepsize(self, a, d, lam = 1, epsilon = .5):
        alpha = 2*lam*d - 2*a*d - 2*lam*(d**2)
        beta = (a**2 + (lam**2)*(d**3) 
                + 2*lam*a*(d**2) + (a**2)*d)
        eta = (sqrt((alpha**2) + 4*beta*(epsilon - d)) + alpha)/(2*beta)
        return(eta)

    def _take_step(self, x, alpha, direction, projection):
        w_new = x + alpha * direction
        subspace, x_new = projection(w_new)
        return(subspace, x_new)
    
    def solve(self, problem, lam=1, epsilon=0.5, x=None):
        """
        Perform optimization over a manifold using a landing field.
        
        Arguments:
            - problem (manifold problem)
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

        A = problem.A
        verbosity = problem.verbosity
        s = problem.sparsity
        r = problem.rank
        n = np.prod(A.shape_input)
        manifold = Stiefel(n, r)
        
        regularizer = lambda x: .25*lam*np.linalg.norm(x.T @ x - np.eye(x.shape[1]),'fro')**2
        regularizer_gradient = lambda x: lam * x @ (x.T @ x - np.eye(x.shape[1]))
        
        objective = lambda x: problem.objective(x)
        

        MAX_ITER_LSEARCH = 100;
        alpha_bar = 1
        tau = 0.5
        beta = 1e-4

        eta = inf
        HTso = Sparsity(s)

        extraiterfields = ['gradnorm', 'regnorm', 'iter_lsearch']
        self._start_optlog(None, extraiterfields)

        objective_value = objective(x)
        if verbosity >= 2:
            print(" iter\t\t   obj. value\t    grad. norm\t    reg. norm\t iter. lsearch")

        stop_reason = None
        iter = 0
        time0 = time.time()
        while True:
            # Calculate new cost, grad and gradnorm
            projected_gradient = manifold.proj(x, problem.gradient(x))
            regularization_gradient = regularizer_gradient(x)

            # Calculate safe step-size
            regnorm = regularizer(x)
            gradnorm = np.linalg.norm(projected_gradient, 'fro')
            
            # Compute the landing field step-direction
            step_direction = -(projected_gradient + lam*regularization_gradient)

            if self.stepsize_type == 'safeguard':
                # Compute the safe-guard value for eta_new
                eta_new = self._safe_stepsize(gradnorm, regnorm, lam=lam, epsilon=epsilon)
                eta = min(eta, eta_new)
                subspace, x_new = self._take_step(x, eta, step_direction, HTso.project)
                iter_lsearch = 1
            elif self.stepsize_type == 'barmijo':
                # Apply line-search with backtracking Armijo
                alpha = alpha_bar
                iter_lsearch = 1
                while True:
                    subspace, x_new = self._take_step(x, alpha, step_direction, HTso.project)
                    s_new = x_new - x
                    objective_value_new = objective(x_new)
                    decrease_cond = (objective_value - objective_value_new >= beta*( - np.dot(s_new.flatten(), step_direction.flatten())))
                    safeguard_cond = (4*regularizer(x_new) <= epsilon)
                    if (decrease_cond and safeguard_cond) or iter_lsearch > MAX_ITER_LSEARCH:
                        break
                    alpha = alpha * tau
                    iter_lsearch = iter_lsearch + 1
            x = x_new
            objective_value = objective(x)
            #epsilon = 4*regularizer(x_new)
            iter = iter + 1
            if verbosity >= 2:
                print("%5d\t%+.16e\t%.8e\t%.8e\t%5d" % (iter, objective_value, gradnorm, regnorm, iter_lsearch))

            if self._logverbosity >= 2:
                self._append_optlog(iter, time0, objective_value, 
                                    gradnorm = gradnorm, 
                                    regnorm = regnorm)

            stop_reason = self._check_stopping_criterion(
                time0, iter=iter, objective_value=objective_value, stepsize=eta, gradnorm=gradnorm)

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

        
        if self._logverbosity <= 0:
            return (subspace, x)
        else:
            self._stop_optlog(stop_reason, iter, time0, objective(x),
                              stepsize=eta, 
                              gradnorm=gradnorm,
                              regnorm=regnorm)
            return (subspace, x, self._optlog)