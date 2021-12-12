import abc
import time


class Solver(metaclass=abc.ABCMeta):
    '''
    Abstract base class setting out template for solver classes.
    Method list:
    '''

    def __init__(self, maxtime=1000, maxiter=1000, minobjective_value=1e-8, minreldecrease=1-1e-3, mingradnorm=1e-8, minstepsize=1e-10,  maxcostevals=float('inf'), logverbosity=1):
        """
        Variable attributes (defaults in brackets):
            - maxtime (1000)
                Max time (in seconds) to run.
            - maxiter (1000)
                Max number of iterations to run.
            - mingradnorm (1e-8)
                Terminate if the norm of the gradient is below this.
            - minstepsize (1e-10)
                Terminate if linesearch returns a vector whose norm is below
                this.
            - maxcostevals (Inf)
                Maximum number of allowed cost evaluations
            - logverbosity (0)
                Level of information logged by the solver while it operates,
                0 is silent, 2 is most information.
        """

        self._maxtime = maxtime
        self._maxiter = maxiter
        self._minobjective_value = minobjective_value
        self._minreldecrease = minreldecrease
        self._mingradnorm = mingradnorm
        self._minstepsize = minstepsize
        self._maxcostevals = maxcostevals
        self._logverbosity = logverbosity
        self._optlog = None

    def __str__(self):
        return type(self).__name__

    @abc.abstractmethod
    def solve(self, problem, x=None):
        '''
        Solve the given :py:class:`pynopt.core.problem.Problem` (starting
        from a _compute_initial_guess if the optional argument x is not
        provided).
        '''
        pass

    def _check_stopping_criterion(self, time0, iter=-1, objective_value=float('inf'), 
                                    reldecrease = 1, gradnorm=float('inf'), 
                                    stepsize=float('inf'), costevals=-1):
        reason = None
        if time.time() >= time0 + self._maxtime:
            reason = ("Terminated - max time reached after %d iterations."
                      % iter)
        elif iter >= self._maxiter:
            reason = ("Terminated - max iterations reached after "
                      "%.2f seconds." % (time.time() - time0))
        elif objective_value < self._minobjective_value:
            reason = ("Terminated - target objective reached after "
                      "%.2f seconds." % (time.time() - time0))
        elif gradnorm < self._mingradnorm:
            reason = ("Terminated - min grad norm reached after %d "
                      "iterations, %.2f seconds." % (
                          iter, (time.time() - time0)))
        elif stepsize < self._minstepsize:
            reason = ("Terminated - min stepsize reached after %d iterations, "
                      "%.2f seconds." % (iter, (time.time() - time0)))
        elif costevals >= self._maxcostevals:
            reason = ("Terminated - max cost evals reached after "
                      "%.2f seconds." % (time.time() - time0))
        return reason

    def _start_optlog(self, solverparams=None, extraiterfields=None):
        ''' Initialize dictionary for logging the iterations and tracking values'''
        if self._logverbosity <= 0:
            self._optlog = None
        else:
            self._optlog = {'solver': str(self),
                            'stoppingcriteria': {'maxtime':
                                                 self._maxtime,
                                                 'maxiter':
                                                 self._maxiter,
                                                 'mingradnorm':
                                                 self._mingradnorm,
                                                 'minstepsize':
                                                 self._minstepsize,
                                                 'maxcostevals':
                                                 self._maxcostevals},
                            'solverparams': solverparams,
                            'final_values': {}
                            }
        
        # If _log_verbosity >= 2 track individual iterations
        if self._logverbosity >= 2:
            self._optlog['iterations'] = {'iteration': [], 
                                            'time': [],
                                            'fx': []}
            if extraiterfields:
                for field in extraiterfields:
                    self._optlog['iterations'][field] = []

    def _append_optlog(self, iteration, time0, fx, **kwargs):
        ''' Append log of iterations and tracking values.'''
        self._optlog['iterations']['iteration'].append(iteration)
        self._optlog['iterations']['time'].append(time.time() - time0)
        self._optlog['iterations']['fx'].append(fx)
        for key in kwargs:
            if kwargs[key] != None:
                self._optlog['iterations'][key].append(kwargs[key])

    def _stop_optlog(self, stop_reason, iteration, time0, fx, **kwargs): 
        self._optlog['stop_reason'] = stop_reason
        self._optlog['final_values'] = {'iteration': iteration,
                                        'fx': fx,
                                        'time': time.time() - time0}
        for key in kwargs:
            if kwargs[key] != None:
                self._optlog['final_values'][key] = kwargs[key]