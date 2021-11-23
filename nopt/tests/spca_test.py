'''
This is to test recovery properties of Sparse PCA algorithm. Let
    D ~ N(0, X_0*thetas*X_0.T + I)
be a sample matrix of N samples in mathbb{R}^n, where $X_0\in\mathrm{St}(n,r)$
such that it is also sparse.

'''

import itertools

import numpy as np

from nopt.tests import GenerateQSMat1
from nopt.transforms import LinearMatrix
from nopt.problems import SparsePrincipalSubspace

def benchmark_gaussian_model(method, dims, ranks, sparsities, num_samples, thetas, seed = 123):
    # Set the random seed for the experiment
    np.random.seed(seed)

    params_list = itertools.product(dims, ranks, sparsities, num_samples, thetas)
    errors = params_list
    for params in params_list:
        n = params[0]
        rank = params[1]
        sparsity = params[2]
        num_samples = params[3]
        theta = params[4]
        
        # Generate the sparse subspace
        subspace0, q0 = GenerateQSMat1((n,rank), sparsity)
        samples = np.random.multivariate_normal(np.zeros((n,)), 
                                                np.eye(n) + theta * q0 @ q0.T, num_samples).T
        A = LinearMatrix(samples @ samples.T / num_samples)

        problem = SparsePrincipalSubspace(A, rank = rank, sparsity=sparsity)

        subspace, x, opt_log = method(problem)

        error = np.linalg.norm(np.abs(q0.T @ x) @ x - q0, 'fro')/np.linalg.norm(q0, 'fro')

