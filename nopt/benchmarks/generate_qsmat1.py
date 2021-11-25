import numpy as np

from nopt.constraints import Sparsity


def GenerateQSMat1(size, sparsity, tol = 1e-16, MAX_ITER = 20, verb = 0):
    """
    Generate low-rank plus sparse matrix
    Attributes:
        - size
            A linear operator going from R^{m\times n} to R^p
            If none given, an identity is assumed
        - sparsity
             sparsity
    """
    tol = 1e-16
    q0 = np.random.normal(0, 1, (size[0], size[1]))
    HTs = Sparsity(sparsity)

    delta = 1
    iter = 1
    while (delta > tol) and (iter <= MAX_ITER):
        q0_new, _ = np.linalg.qr(q0)
        subspace, q0_new = HTs.project(q0_new)
        delta = np.linalg.norm(q0_new - q0, 'fro')/np.linalg.norm(q0, 'fro')
        q0[:] = q0_new[:]
        iter = iter + 1
    if verb >= 1:
        print(delta)
        print(np.max(np.abs(q0.transpose() @ q0) - np.eye(size[1])))
    return (subspace, q0)