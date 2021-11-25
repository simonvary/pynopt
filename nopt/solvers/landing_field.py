import time
from math import inf, sqrt
import numpy as np

from pymanopt.manifolds import Stiefel

from nopt.constraints.sparse_oblique import SparseOblique
from nopt.constraints import FixedRank, Sparsity

from nopt.solvers.solver import Solver
import pdb

class LandingField(Solver):

    def __init__(self, linesearch='normalized', *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def _compute_stepsize(self, gradient, subspace, A, constraint):
        gradient_proj = constraint.project_subspace(gradient, subspace)
        a = np.linalg.norm(gradient_proj)**2
        b = np.linalg.norm(A.matvec(gradient_proj))**2
        return a/b

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

        eta = inf
        HTso = SparseOblique(s)

        objective_value = objective(x)
        if verbosity >= 2:
            print(" iter\t\t   obj. value\t    grad. norm\t    reg. norm")

        self._start_optlog()
        stop_reason = None
        iter = 0
        time0 = time.time()
        #pdb.set_trace()
        while True:
            # Calculate new cost, grad and gradnorm
            projected_gradient = manifold.proj(x, problem.gradient(x))
            regularization_gradient = regularizer_gradient(x)

            # Calculate safe step-size
            regnorm = regularizer(x)
            gradnorm = np.linalg.norm(projected_gradient, 'fro')
            eta_new = self._safe_stepsize(gradnorm, regnorm, lam=lam, epsilon=epsilon)
            eta = min(eta, eta_new)
            
            step_direction = projected_gradient + lam*regularization_gradient
            subspace, x_new = self._take_step(x, eta, -step_direction, HTso.project_quasi)
            
            x = x_new
            objective_value = objective(x)

            iter = iter + 1
            if verbosity >= 2:
                print("%5d\t%+.16e\t%.8e\t%.8e" % (iter, objective_value, gradnorm, regnorm))

            if self._logverbosity >= 2:
                self._append_optlog(iter, objective_value, xdist = None)

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
            self._stop_optlog(x, objective(x), stop_reason, time0,
                              stepsize=eta, gradnorm=gradnorm,
                              iter=iter)
            return (subspace, x, self._optlog)