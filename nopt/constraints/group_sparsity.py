import numpy as np

from nopt.constraints.constraint import Constraint

class GroupSparsity(Constraint):
    """
    Projections of tensors based on group sparsity
    """

    def __init__(self, groups, ks):
        '''
            ind_list is a list of lists
            k is a list of sparsities for each index group
        '''
        self.groups = groups
        self.ks = ks

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
        
        _x = x.copy()
        inds = np.zeros_like(x,dtype=bool)

        for (group, k) in zip(groups, ks):
            x_group = x[group]
            _x_group = _x[group]
            inds_group = inds[group]
            ind = np.argpartition(np.abs(x_group), -k, axis=None)[-k:]
            ind = np.unravel_index(ind, _x.shape)
            ind_del = np.ones(len(group), dtype=bool)
            ind_del[ind] = False
            _x_group[ind_del] = 0
            inds_group[ind[0]] = True
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