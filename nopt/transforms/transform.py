import abc

class Transform(metaclass=abc.ABCMeta):
    '''
    Abstract base class setting out template for transform classes.
    '''

    def __init__(self, shape_input = None, shape_output = None, verbosity = 0):
        """
        Variable attributes (defaults in brackets):
            - verbosity (0)
                Level of information logged by the solver while it operates,
                0 is silent, 2 ist most information.
            - shape_input (array)
                Shape of the numpy array input
            - shape_output (array)
                Shape of the numpy array output
        """
        self._verbosity = verbosity
        self.shape_input = shape_input
        self.shape_output = shape_output

    def __str__(self):
        return type(self).__name__

    @abc.abstractmethod
    def matvec(self, x):
        '''
        Forward operation by the transform acting on x.
        '''
        # Could have a general check for the shape of the input
        # could have a general function for changing shapes
        pass