import abc
from scipy.sparse.linalg import LinearOperator

class TensorLinearOperator(metaclass=abc.ABCMeta):
    """
    Linear transform based on a numpy array
        Variable attributes (defaults in brackets):
            - verbosity (0)
                Level of information logged by the solver while it operates,
                0 is silent, 2 ist most information.
            - shape_input (array)
                Shape of the numpy array input
            - shape_output (array)
                Shape of the numpy array output
    """

    def __init__(self, shape: tuple, linear_operator=None):
        self.shape = shape
        if linear_operator:
            self.linear_operator = linear_operator
        else:
            class Id(object):
                def __init__(self):
                    pass

                def matvec(self,x):
                    return x

                def rmatvec(self,y):
                    return y        
            self.linear_operator = Id()

    # Function to apply the transform.
    def matvec(self, x):
        y = self.linear_operator.matvec(x.flatten())
        return y

    # Function to apply adjoint/backward transform.
    def rmatvec(self, y):
        x = self.linear_operator.rmatvec(y).reshape(self.shape)
        return x#.reshape(self.shape_input)
        # still have to add the reshapes
