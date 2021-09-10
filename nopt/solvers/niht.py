import time
import numpy as np

from nopt.solvers.solver import Solver


class NIHT(Solver):

    def __init__(self, linesearch='normalized', *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def _compute_stepsize(self, gradient, subspace, A, constraint):
        gradient_proj = constraint.project_subspace(gradient, subspace)
        a = np.linalg.norm(gradient_proj)**2
        b = np.linalg.norm(A(gradient_proj))**2
        return a/b

    def _compute_initial_guess(self, A, b, constraint):
        w = A.adjoint(b)
        T_k, x = constraint.project(w)
        return (T_k, x)

    def solve(self, problem, x=None):
        """
        Perform optimization using gradient descent with linesearch.
        This method first computes the gradient (derivative) of obj
        w.r.t. arg, and then optimizes by moving in the direction of
        steepest descent (which is the opposite direction to the gradient).
        Arguments:
            - problem
                Pymanopt problem setup using the Problem class, this must
                have a .manifold attribute specifying the manifold to optimize
                over, as well as a cost and enough information to compute
                the gradient of that cost.
            - x=None
                Optional parameter. Starting point. If none
                then a starting point will be randomly generated.
            - reuselinesearch=False
                Whether to reuse the previous linesearch object. Allows to
                use information from a previous solve run.
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
        """

        # Check the problem is LinearLeastSquares type
        constraint = problem.constraint
        objective = problem.objective
        A = problem.A
        b = problem.b
        verbosity = problem.verbosity

        if x is None:
            subspace, x = self._compute_initial_guess(A, b, constraint)
        else:
            subspace, _ = constraint.project(x)

        if verbosity >= 2:
            print(" iter\t\t   obj. value\t    grad. norm")

        self._start_optlog()
        stop_reason = None
        iter = 0
        time0 = time.time()

        while True:
            # Calculate new cost, grad and gradnorm
            objective_value = objective(x)
            iter = iter + 1

            grad = A.adjoint(A(x) - b)
            gradnorm = np.linalg.norm(grad, 2)
            alpha = self._compute_stepsize(grad, subspace, A, constraint)
            w = x - alpha * grad
            subspace, x = constraint.project(w)


            if verbosity >= 2:
                print("%5d\t%+.16e\t%.8e" % (iter, objective_value, gradnorm))

            if self._logverbosity >= 2:
                self._append_optlog(iter, x, objective_value, gradnorm=gradnorm)

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
            self._stop_optlog(x, objective(x), stop_reason, time0,
                              stepsize=alpha, gradnorm=gradnorm,
                              iter=iter)
            return x, self._optlog