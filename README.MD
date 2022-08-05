# Pynopt

This repository includes simple numpy implementations for some non-smooth, non-convex optimisation problems arising in data processing, such as
* compressed sensing
* matrix completion/sensing
* robust low-rank estimation (Robust PCA)

The goal is to write a algorithms for a unified treatment of the above to be used in a number of applications, such as [video seperation](https://github.com/simonvary/pynopt/blob/master/examples/example_video.ipynb), multispectral imaging, [exoplanet detection](https://github.com/simonvary/pynopt/blob/master/examples/example_exoplanet.ipynb), [image inpainting](https://github.com/simonvary/pynopt/blob/master/examples/example_image_inpainting.ipynb), and others.


## Problem classes

### LinearProblem
Solve

$$ \min_{x} \big\| A(x) - b \big\|_2\qquad \mathrm{s.t.}\quad x \in C$$

where $C$ can be a sparse set or a fixed rank set, and $A$ is a linear operator from [pylops](https://github.com/PyLops/pylops) or [scipy's LinearOperator class](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html).

Solvers: NIHT

### LinearProblemSum
Solve the linear combination

$$ \min_{x} \big\| \sum_{i}^{n}A_i(x_i) - b \big\|_2\qquad \mathrm{s.t.}\quad x_i \in C_i$$

Solvers: NAHT

## Requirements
- numpy (1.21.3)
- pandas (1.3.4)
- scikit-learn (1.0.1)


## Citation
```
@misc{Tanner2020compressed,
  author = {Tanner, Jared and Vary, Simon},  
  title = {Compressed sensing of low-rank plus sparse matrices},
  url = {https://arxiv.org/abs/2007.09457},
  publisher = {arXiv},
  year = {2020},
}
```

