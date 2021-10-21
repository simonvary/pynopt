import numpy as np


def GenerateLSMat1(size, rank, sparsity, const):
    """
    Generate low-rank plus sparse matrix
    Attributes:
        - size
            A linear operator going from R^{m\times n} to R^p
            If none given, an identity is assumed
        - rank
            A right-hand side in R^p
        - sparsity
            A callable which takes an element of R^p and returns a real number.
        - const
    """
    P = np.random.normal(0, 1, (size[0], rank))
    Q = np.random.normal(0, 1, (size[1], rank))
    L = P @ Q.transpose()
    S = np.zeros((size[0], size[1]))
    S_ind = np.random.choice(S.size, size=sparsity, replace=False)
    S[np.unravel_index(S_ind, S.shape)] = (2*np.random.rand(sparsity)-1)*const* max(np.mean(abs(L)), 0.2)
    return (L,S)