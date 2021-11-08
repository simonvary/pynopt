


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
        self._oldf0 = None

    def search(self, objective, x, direction, projection):
        """
        Function to perform backtracking line-search.
        Arguments:
            - objective
                objective function to optimise
            - x
                starting point on the manifold
            - projection
                projection on the constraint)
        Returns:
            - stepsize
                norm of the vector retracted to reach newx from x
            - newx
                next iterate suggested by the line-search
        """