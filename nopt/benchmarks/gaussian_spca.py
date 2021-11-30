'''
This is to test recovery properties of Sparse PCA algorithm. Let
    D ~ N(0, X_0*thetas*X_0.T + I)
be a sample matrix of N samples in mathbb{R}^n, where $X_0\in\mathrm{St}(n,r)$
such that it is also sparse.

'''

import itertools

import pandas as pd  
import numpy as np

from nopt.benchmarks import GenerateQSMat1
from nopt.transforms import LinearMatrix
from nopt.problems import SparsePrincipalSubspace

class GaussianSPCA(object):
    """
        Perform a set of benchmarks on syntetically generated data.
        The model:
            X_i~Normal(0,1, eye(n) + theta*q_0 @ q_0.T )
            N-samples
        Recover q0 from X_i's
        Arguments:
            - problem (PrincipalSubspace)
            - x=None
                Optional parameter. Starting point. If none
                then a starting point will be randomly generated.
            - stepsize_type = barmijo
                Whether to reuse the previous linesearch object. Allows to
                use information from a previous solve run.
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
    """

    def __init__(self, verb = 1):
        self.verb = verb
        return None

    def single_test(self, solve_func, params, track_fields = None, seed = None):
        if seed is not None:
            np.random.seed(seed)
        
        if track_fields is None:
            track_fields = ['fx', 'iteration', 'time', 
                            'dist_fx_true', 'dist_x_true']

        n = params['n']
        rank = params['rank']
        alpha = params['alpha']
        num_samples = params['num_samples']
        theta = params['theta']

        sparsity = np.round(alpha*n*rank).astype(int).item()

        # Generate the sparse subspace
        if self.verb >= 2:
            print('Generating q0.')
        subspace0, q0 = GenerateQSMat1((n,rank), sparsity)
        # Sample the distribution
        if self.verb >= 2:
            print('Sampling the distribution q0.')
        samples = np.random.multivariate_normal(np.zeros((n,)), 
                                                    np.eye(n) + theta * q0 @ q0.T, num_samples).T
        # Construct the correlation matrix
        if self.verb >= 2:
            print('Constructing the SparseSubspace problem.')
        A = LinearMatrix(samples @ samples.T / num_samples)
        problem = SparsePrincipalSubspace(A, rank = rank, sparsity=sparsity, 
                                            verbosity = 0, x_true = q0)
        
        if self.verb >= 2:
            print('Solving the problem.')
        subspace, x, opt_log = solve_func(problem)

        track_values = {}
        for key in track_fields:
            track_values[key] = opt_log['final_values'][key]

        if self.verb >= 1:
            print('Problem solved in %d iterations.' % len(opt_log['iterations']['iteration']))
        return opt_log, track_values, problem


    def grid_test(self, solver_func, params_list, repetitions = 1, track_fields = None, seed = None):
        if seed is not None:
            np.random.seed(seed)

        if track_fields is None:
            track_fields = ['fx', 'iteration', 'time', 
                            'dist_fx_true', 'dist_x_true']

        test_results = pd.DataFrame(columns=list(params_list.keys()) + track_fields)

        for param_value in itertools.product(*params_list.values()):
            param_dict = dict(zip(params_list.keys(), param_value))
            _, track_values, _ = self.single_test(solver_func, param_dict, track_fields = track_fields, seed = None)

            for field in track_fields:
                param_dict[field] = track_values[field]

            test_results = test_results.append(param_dict, ignore_index=True)

        return(test_results)


