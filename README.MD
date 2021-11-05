

# Pynopt

This repository includes simple implementations for certain non-smooth, non-convex optimisation problems arising in data processing. Exemplar applications include:
- compressed sensing
- matrix completion/sensing
- robust low-rank estimation (Robust PCA)
- sparse prinicipal subspace estimation (Sparse PCA)

The main goal is to **stop rewritting the same projected gradient method** all over again and to have a **unified** treatment of linear operators, e.g. curvelets, and a set of quick tests to quickly assess efficiancy of the algorithms.

Currently, there are the following solvers:
- Normalized Iterative Hard Thresholding (LinearProblem)
- Normalized Alternating Hard Thresholding (LinearProblemSum)
- Oblique Sparse PCA (SparsePrincipalSubspace)

Some things to keep in mind:
- use the syntax close to scipy.sparse.linalg.LinearOperator for Transforms, allowing for
    - implementation of scalable fast operators, e.g. FJLT, etc.
    - readily availibility to use [curvelops](https://github.com/PyLops/curvelops), wavelets, etc.
- use syntax close to the PyProximal for non-smooth proximal operators?
- use syntax close to scikit.learn for solvers?
