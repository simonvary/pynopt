import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

import pylops
from matplotlib import colors
import matplotlib as mpl

import scipy.io as sio

from nopt.transforms import *
from nopt.constraints import *
from nopt.problems import *
from nopt.solvers import *
import h5py
from sklearn.metrics.cluster import adjusted_rand_score


def load_hdf5(filename):
    f = h5py.File(filename,'r+')
    return(f['exchange']['data'][()])

def load_mat_array(filename):
    matstruct = sio.loadmat(filename)
    x = matstruct['A_proj'].todense()
    _,mask = np.where(matstruct['Omega'].todense().flatten())
    return (x, mask)

def load_mat_array_full(filename):
    matstruct = sio.loadmat(filename)
    x = matstruct['data']
    return x



experiment_name = 'ds4'

with open(experiment_name + '.pickle', 'rb') as handle:
    results = pickle.load(handle)

sparsity_ratios = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

tensor_shape = results['full']['tensor_shape']

x = results['full']['x'].reshape(tensor_shape)
x_mat = x.reshape(tensor_shape[0], np.prod(tensor_shape[-2:]))


labels_full = results['full']['labels']
for key_delta, d_exper in results.items():
    if key_delta != 'full':
        print('Delta: '+str(key_delta))
        rhos_s = 'Rhos: '
        sparsities = 'Sparsities: '
        errors_lrps = 'Errors LRPS: '
        aris = '1-ARI: '
        for key_rho_s, s_exper in d_exper.items():
            labels_subsampled = s_exper['labels']
            rhos_s = rhos_s + str(key_rho_s) + ', '
            sparsity = "%d" % np.round(key_rho_s * np.prod(results['full']['tensor_shape']))
            sparsities = sparsities + sparsity + ', '
            errors_lrps = errors_lrps + ("%.3f" % s_exper['error_lrps']) + ', '
            ari = adjusted_rand_score(labels_full, labels_subsampled)
            aris = aris + ("%.3f" % (1-ari)) + ', '
        print(rhos_s)
        print(sparsities)
        print(errors_lrps)
        print(aris)