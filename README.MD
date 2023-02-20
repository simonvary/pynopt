# Pynopt

Simple numpy implementations for some non-smooth, non-convex optimisation problems arising in data processing, such as
* compressed sensing
* matrix completion/sensing
* robust low-rank estimation (Robust PCA)

The goal is to write a algorithms for a unified treatment of the above to be used in a number of applications, such as [video seperation](https://github.com/simonvary/pynopt/blob/master/examples/example_video.ipynb), [multispectral imaging](https://github.com/simonvary/pynopt/blob/master/examples/example_multispectral_image.ipynb), [exoplanet detection](https://github.com/simonvary/pynopt/blob/master/examples/example_exoplanet.ipynb), [image inpainting](https://github.com/simonvary/pynopt/blob/master/examples/example_image_inpainting.ipynb), and others.


## Problem classes

### LinearProblem
Solve

$$ \min_{x} \big\lVert  A(x) - b \big\rVert_2\qquad \mathrm{s.t.}\quad x \in C$$

where $C$ can be a sparse set or a fixed rank set, and $A$ is a linear operator from [pylops](https://github.com/PyLops/pylops) or [scipy's LinearOperator class](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html).

Solvers: NIHT

### LinearProblemSum
Solve the linear combination

$$ \min_{x_i} \big\lVert A_1(x_1) + A_2(x_2)  - b \big\rVert_2\qquad \mathrm{s.t.}\quad x_i \in C_i$$

Solvers: NAHT

## Requirements
- numpy (1.21.3)
- pandas (1.3.4)
- scikit-learn (1.0.1)


## Citation
```
@article{Tanner2023Compressed,
    title = {Compressed sensing of low-rank plus sparse matrices},
    author = {Jared Tanner and Simon Vary},
    journal = {Applied and Computational Harmonic Analysis},
    volume = {64},
    pages = {254-293},
    year = {2023},
    issn = {1063-5203},
    doi = {https://doi.org/10.1016/j.acha.2023.01.008},
    url = {https://www.sciencedirect.com/science/article/pii/S106352032300009X}
}
```

