

class Identity(object):
    """
    Linear transform based on a numpy array
    """
    def __init__(self):
        pass
    
    def matvec(self, x):
        return x
    
    def rmatvec(self, y):
        return y