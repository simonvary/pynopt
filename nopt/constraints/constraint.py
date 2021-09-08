


# Abstract class
class Constraint(object):
    def __init__(self, k):
        pass

    def project(self, x, k=None):
        pass

    def project_subspace(self, x, subspace):
        pass