


class LineSearchBackTracking:
    """
    Back-tracking line-search based on linesearch.m in the manopt MATLAB package
    """

    def __init__(self, initial_stepsize = 1, suff_decrease = 1e-4, optimism = 2, contraction_factor = 0.5, MAX_ITER = 100):
        self.initial_stepsize = initial_stepsize
        self.optimism = optimism
        self.contraction_factor = contraction_factor
        self.suff_decrease = suff_decrease
        self.MAX_ITER = MAX_ITER

        # Possible to reuse old values as a start
        self._oldf0 = None

    def search(self, objective, take_step, alpha_0):
        """
        Function to perform backtracking line-search.
        Arguments:
            - function_line
                Objective function with 1 parameter 
            - x
                starting point
            - projection
                projection on the constraint)
        Returns:
            - stepsize
                norm of the vector retracted to reach newx from x
            - newx
                next iterate suggested by the line-search
        """

        alpha = self.alpha_bar
        iter_lsearch = 1
        while True:
            subspace, x_new = self._take_step(x, alpha, -grad, HTso.project)
            s_new = x_new - x
            objective_value_new = objective(x_new)
            if objective_value - objective_value_new >= beta*( - np.dot(s_new.flatten(), grad.flatten())) or iter_lsearch > MAX_ITER_LSEARCH:
                break
            alpha = alpha * tau
            iter_lsearch = iter_lsearch + 1
