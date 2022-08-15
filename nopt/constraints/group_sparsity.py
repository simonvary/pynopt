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

    def project(self, x, k=None):
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
        if k is None:
            k = self.k
        _x = x.copy()

        if isinstance(k, int):
            ind = np.argpartition(np.abs(x), -k, axis=None)[-k:]
            ind = np.unravel_index(ind, _x.shape)
            ind_del = np.ones_like(_x, dtype=bool)
            ind_del[ind] = False
            _x[ind_del] = 0
        else:
            ind_tmp = np.argpartition(np.abs(_x), kth=-k, axis=0)
            ind_keep = np.ones_like(_x, dtype=bool)
            for i in range(x.shape[1]):
                ind_keep[ind_tmp[:-k[i],i],i] = False
            _x[np.logical_not(ind_keep)] = 0
            ind = np.where(ind_keep)
        return np.sort(ind[0]), _x

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