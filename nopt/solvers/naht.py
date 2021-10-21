import time
import numpy as np

from nopt.solvers.solver import Solver


class NAHT(Solver):
    """r
    Normalized Alternating Hard Thresholding 
    
    """

    def __init__(self, linesearch='normalized', *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def _compute_stepsize(self, gradient, subspace, A, constraint):
        gradient_proj = constraint.project_subspace(gradient, subspace)
        a = np.linalg.norm(gradient_proj)**2
        b = np.linalg.norm(A.matvec(gradient_proj))**2
        return a/b

    def _compute_initial_guess(self, A, b, constraints):
        w = A.rmatvec(b)
        T_1, x1 = constraints[0].project(w)
        T_2, x2 = constraints[1].project(w - x1) 
        return [[T_1, T_2], [x1, x2]]

    def solve(self, problem, x=None):
        """
        Normalized Alternating Hard Thresholding for a recovery of an additive combination of two nonconvex sets 
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
        constraints = problem.constraints
        objective = problem.objective
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
            objective_value = objective(x[0] + x[1])
            iter = iter + 1

            # gradient step for the first component
            grad = A.rmatvec(A.matvec(x[0] + x[1]) - b)
            gradnorm = np.linalg.norm(grad, 2)
            alpha = self._compute_stepsize(grad, subspaces[0], A, constraints[0])
            v = x[0] - alpha * grad
            temp_subspace, temp_x = constraints[0].project(v)
            subspaces[0] = temp_subspace
            x[0] = temp_x

            # gradient step for the second component
            grad = A.rmatvec(A.matvec(x[0] + x[1]) - b)
            gradnorm = np.linalg.norm(grad, 2)
            alpha = self._compute_stepsize(grad, subspaces[1], A, constraints[1])
            w = x[1] - alpha * grad
            temp_subspace, temp_x = constraints[1].project(w)
            subspaces[1] = temp_subspace
            x[1] = temp_x

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