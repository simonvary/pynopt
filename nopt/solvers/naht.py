import time
import numpy as np

from nopt.solvers.solver import Solver

class NAHT(Solver):
    """r
    Normalized Alternating Hard Thresholding 

    https://arxiv.org/pdf/2007.09457.pdf

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
        w1 = A[0].rmatvec(b)
        T_1, x1 = constraints[0].project(w1)
        w2 = A[1].rmatvec(A[0].matvec(x1)-b)
        T_2, x2 = constraints[1].project(w2) 
        return [[T_1, T_2], [x1, x2]]

    def solve(self, problem, x0=None):
        """
        Normalized Alternating Hard Thresholding for a recovery of an additive combination of two nonconvex sets 
        Arguments:
            - problem (LinearProblemSum)
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
        
        verbosity = self._verbosity

        # x is now a tuple x = (x1, x2)

        if x0 is None:
            subspaces, x = self._compute_initial_guess(A, b, constraints)
        else:
            x = x0.copy()
            subspaces = [None, None]
            subspaces[0], _ = constraints[0].project(x[0])
            subspaces[1], _ = constraints[1].project(x[1])

        if verbosity >= 2:
            print(" iter\t\t   obj. value\t    grad. norm")

        self._start_optlog()
        stop_reason = None
        iter = 0
        time0 = time.time()

        while True:
            # Calculate new cost, grad and gradnorm
            # objective_value = objective(x[0] + x[1])
            iter = iter + 1

            # gradient step for the first component
            grad = A[0].rmatvec(A[0].matvec(x[0]) + A[1].matvec(x[1]) - b)
            gradnorm = np.linalg.norm(grad, 2)
            alpha = self._compute_stepsize(grad, subspaces[0], A[0], constraints[0])
            v = x[0] - alpha * grad
            temp_subspace, temp_x = constraints[0].project(v)
            subspaces[0] = temp_subspace
            x[0] = temp_x

            # gradient step for the second component
            grad = A[1].rmatvec(A[0].matvec(x[0]) + A[1].matvec(x[1]) - b)
            gradnorm = np.linalg.norm(grad, 2)
            alpha = self._compute_stepsize(grad, subspaces[1], A[1], constraints[1])
            w = x[1] - alpha * grad
            temp_subspace, temp_x = constraints[1].project(w)
            subspaces[1] = temp_subspace
            x[1] = temp_x

            objective_value = objective(x)
            running_time = time.time() - time0

            if verbosity >= 2:
                print("%5d\t%+.16e\t%.8e" % (iter, objective_value, gradnorm))

            if self._logverbosity >= 2:
                self._append_optlog(iter, running_time, objective_value, xdist = None) # gradnorm=gradnorm

            stop_reason = self._check_stopping_criterion(
                running_time, iter=iter, objective_value=objective_value, stepsize=alpha, gradnorm=gradnorm)

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

        
        if self._logverbosity <= 0:
            return x
        else:
            self._stop_optlog(iter, objective(x), stop_reason, running_time,
                              stepsize=alpha, gradnorm=gradnorm)
            return x, self._optlog