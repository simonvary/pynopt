# Pynopt

This repository includes simple numpy implementations for some non-smooth, non-convex optimisation problems arising in data processing, such as

* compressed sensing
* matrix completion/sensing
* robust low-rank estimation (Robust PCA)

The goal is to write a algorithms for a unified treatment of the above to be used in a number of applications, such as [video seperation](), [multispectral imaging](), exoplanet detection, image inpainting, and others.


## Problem classes

### LinearProblem

$$ \min_{x} \big\| A(x) - b \big\|_2\qquad \mathrm{s.t.}\quad x \in C$$

where $$C$$ can be one of the following

- sparse set
- fixed rank set

and $$A$$ is a linear operator from [pylops](https://github.com/PyLops/pylops) or [scipy's LinearOperator class](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html).


Solvers: NIHT

### LinearProblemSum

$$ \min_{x} \big\| \sum_{i}A_i(x_i) - b \big\|_2\qquad \mathrm{s.t.}\quad x_i \in C_i$$

Solvers: NAHT

## Algorithms

### NIHT

### NAHT


### Requirements
- numpy (1.21.3)
- pandas (1.3.4)
- scikit-learn (1.0.1)
