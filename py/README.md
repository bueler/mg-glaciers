# mg-glaciers/py/

These Python codes, examples of applying the multilevel constraint decomposition of Tai (2003) on obstacle problems in glaciology, support the two-part paper [_Geometric multigrid for glacier modeling_](../paper/).

## partI/

We solve three one-dimensional problems using basic Python, along with standard libraries including [NumPy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/):

  1. a classical obstacle problem for the Poisson equation (Laplace operator) on the interval [0,1],
  2. the corresponding obstacle problem for the p-Laplacian operator, and
  3. the steady and implicit time-stepping obstacle problem for the shallow ice approximation (SIA) on an interval [0,L].

All of these are solved with a from-scratch implementation of the piecewise-linear (P1) finite element method.

## partII/

UNDER CONSTRUCTION

These programs solve the steady and implicit time-stepping geometry-evolution for glaciers using Stokes dynamics.  These programs use the [Firedrake](https://www.firedrakeproject.org/) library, but we continue to default to P1 elements.
