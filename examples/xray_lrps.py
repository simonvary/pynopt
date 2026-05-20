import sys
sys.path.append("../")

import pickle


import numpy as np
import matplotlib.pyplot as plt
import cv2

import pylops

from nopt.transforms import *
from nopt.constraints import *
from nopt.problems import *
from nopt.solvers import *
import h5py

from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
import scipy.io as sio


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



experiment_name = 'ds5'
if experiment_name == 'ds1':
    filenames = { 'd001' : '../data/dataset_DS1_subsamp_0.01.mat', 
                'd005' : '../data/dataset_DS1_subsamp_0.05.mat', 
                'd020' : '../data/dataset_DS1_subsamp_0.20.mat'}
    filename_full = '../data/DS1.mat'
    rank = 6
elif experiment_name == 'ds2':
    filenames = {'d005' : '../data/dataset_DS2_subsamp_0.05.mat', 
                'd020' : '../data/dataset_DS2_subsamp_0.20.mat'}
    filename_full = '../data/DS2.mat'
    rank = 5
elif experiment_name == 'ds3':
    filenames = { 'd001' : '../data/dataset_DS3_subsamp_0.01.mat', 
                'd005' : '../data/dataset_DS3_subsamp_0.05.mat', 
                'd020' : '../data/dataset_DS3_subsamp_0.20.mat'}
    filename_full = '../data/DS3.mat'
    rank = 4
elif experiment_name == 'ds4':
    filenames = { 'd001' : '../data/dataset_DS4_subsamp_0.01.mat', 
                'd002' : '../data/dataset_DS4_subsamp_0.02.mat', 
                'd005' : '../data/dataset_DS4_subsamp_0.05.mat', 
                'd015' : '../data/dataset_DS4_subsamp_0.15.mat', 
                'd020' : '../data/dataset_DS4_subsamp_0.20.mat'}
    filename_full = '../data/DS4.mat'
    rank = 7
elif experiment_name == 'ds5':
    filenames = { 'd005' : '../data/dataset_DS5_subsamp_0.05.mat', 
                'd020' : '../data/dataset_DS5_subsamp_0.20.mat'}
    filename_full = '../data/DS5.mat'
    rank = 5

sparsity_ratios = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


X_full = load_mat_array_full(filename_full)
X_full_norm = np.linalg.norm(X_full)
X_full = X_full / X_full_norm
X_full_mat = X_full.reshape([X_full.shape[0], np.prod(X_full.shape[-2:])])

m = X_full_mat.shape[0]
n = X_full_mat.shape[1]

HTr = FixedRank(rank, (m,n))
_, X_full_low_rank = HTr.project(X_full_mat)
X_full_low_rank = X_full_low_rank.reshape(m,n)
lr_error = np.linalg.norm(X_full_low_rank - X_full_mat) / np.linalg.norm(X_full_mat)
print('Rank r='+str(rank)+ ' projection error: '+str(lr_error))

_, _, vh = np.linalg.svd(X_full_low_rank)
v = (vh.T)[:,:rank]
scaler = StandardScaler(with_std=False)
scaler.fit(v)
u = scaler.transform(v)
kmeans = KMeans(n_clusters=rank, random_state=0, n_init="auto").fit(v)


results = {}
results['full'] = {
    'x' : X_full_mat,
    'x_full_norm' : X_full_norm,
    'x_lr' : X_full_low_rank,
    'rank' : rank,
    'error_lr' : lr_error,
    'labels' : kmeans.labels_,
    'mat_shape' : [m,n],
    'tensor_shape' : X_full.shape
}

for key, filename in filenames.items():
    X, mask = load_mat_array(filename)
    X = X / X_full_norm

    delta = len(mask) / np.prod(X.shape)
    results[key] = {}

    for rho_s in sparsity_ratios:
        p = round(delta*m*n)
        s = round(rho_s*m*n)
        print(str(rank) +','+ str(s))

        HTs = Sparsity(s)

        # Prepare transform and right-hand side
        P_omega = EntryWise(mask, m*n)
        b = P_omega.matvec(X_full_mat.flatten())

        solver = NAHT(logverbosity = 2, maxiter = 2000, verbosity = 2, minreldecrease = 1-1e-5, minreldecrease_shift = 50)
        problem = LinearProblemSum(P_omega, b, (HTr, HTs))

        x, opt_log = solver.solve(problem)
        x_l = x[0].reshape(m, n)
        x_s = x[1].reshape(m, n)

        lrps_error = np.linalg.norm(x_l +x_s - X_full_mat) / np.linalg.norm(X_full_mat)

        _, _, vh = np.linalg.svd(x_l)
        v = (vh.T)[:,:rank]
        scaler = StandardScaler(with_std=False)
        scaler.fit(v)
        v = scaler.transform(v)
        kmeans = KMeans(n_clusters=rank, random_state=0, n_init="auto").fit(v)

        out = {
            'x_recon' : [x_l, x_s],
            'opt_log' : opt_log,
            'delta' : delta,
            'error_lrps' : lrps_error,
            'rho_s' : rho_s,
            'labels' : kmeans.labels_,
        }

        results[key][rho_s] = out

        with open(experiment_name + '.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)