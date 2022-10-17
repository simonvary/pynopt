import numpy as np

from nopt.constraints.constraint import Constraint

class GroupSparsity(Constraint):
    """
    Projections of tensors based on group sparsity
    """

    def __init__(self, groups, ks, positive = False):
        '''
            ind_list is a list of lists
            k is a list of sparsities for each index group
        '''
        self.groups = groups
        if isinstance(ks, int):
            self.ks = [ks] * len(groups)
        else:
            self.ks = ks
        self.positive = positive

    def project(self, x, groups=None, ks=None):
        """
        Keep only k largest entries of x.
        Parameters
        ----------
        x : numpy array
            Numpy array to be thresholded
        ks : array of ints
            Numbers of largest entries in absolute value to keep in each of the sparsity groups
        Notes
        -----
        """
        if groups is None:
            groups = self.groups
        if ks is None:
            ks = self.ks
        
        if isinstance(ks, int):
            self.ks = [ks] * len(groups)
        else:
            self.ks = ks

        _x = np.zeros_like(x)
        inds = np.zeros_like(x,dtype=bool)


        for (group, k) in zip(groups, ks):
            x_group = x[group]
            _x_group = _x[group]
            inds_group = inds[group]
            if self.positive:
                ind = np.argpartition(x_group, -k, axis=None)[-k:]
            else:
                ind = np.argpartition(np.abs(x_group), -k, axis=None)[-k:]
            #ind = np.unravel_index(ind, _x.shape)
            #ind_del = np.ones_like(x_group, dtype=bool)
            #ind_del[ind] = False
            _x_group[ind] = x_group[ind]
            inds_group[ind] = True
            _x[group] = _x_group
            inds[group] = inds_group

        return inds, _x

    def project_subspace(self, x, ind):
        """
        Keeps only parameters at specified indices setting others to zero
        ----------
        x : numpy array
            Numpy array to be projected
        ind : int
            where to keep entries
        """
        # Check if support is of size k ? 
        _x = x.copy()
        ind_del = np.ones(x.shape, dtype=bool)
        ind_del[ind] = False
        _x[ind_del] = 0
        return _x