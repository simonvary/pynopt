

# Pynopt StructOpt

This repository includes simple implementations for certain non-smooth, non-convex optimisation problems arising in data processing. Exemplar applications include:
- compressed sensing
- matrix completion/sensing
- robust low-rank estimation (Robust PCA)
- sparse prinicipal subspace estimation (Sparse PCA)

The main goal is to **stop rewritting the same proximal gradient method** all over again and to have a **unified** treatment of linear operators, e.g. curvelets, and a set of quick benchmarks to quickly assess efficiancy of the algorithms.

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

### Requirements
- numpy (1.21.3)
- pandas (1.3.4)
- pymanopt (0.2.5)
- scikit-learn (1.0.1)
