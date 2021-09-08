import abc
import time

class Transform(metaclass=abc.ABCMeta):
    '''
    Abstract base class setting out template for transform classes.
    '''

    def __init__(self, longverbosity = 0):
        """
        Variable attributes (defaults in brackets):
            - logverbosity (0)
                Level of information logged by the solver while it operates,
                0 is silent, 2 ist most information.
        """
        self._verbosity = longverbosity

    def __str__(self):
        return type(self).__name__

    @abc.abstractmethod
    def __call__(self, x):
        '''
        Forward operation by the transform acting on x.
        '''
        pass