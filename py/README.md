# mg-glaciers/py/

There are two examples of applying the multilevel constraint decomposition of Tai (2003) on obstacle problems.

## 1D/

We solve the simplest possible problem, a classical obstacle problem for the Poisson equation (Laplace operator) on the interval [0,1], using a from-scratch implementation of the piecewise-linear (P1) finite element method.  These Python programs need only standard libraries and Numpy.

## 2D/

We solve the shallow ice approximation on planar domains.  These Python programs use the Firedrake library and P1 elements.
