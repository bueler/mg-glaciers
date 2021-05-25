# mg-glaciers/py/

These Python examples apply the multilevel constraint decomposition of Tai (2003) on obstacle problems in glaciology.  They support the paper [_Geometric multigrid for glacier modeling_](../paper/).

We solve three one-dimensional problems using basic Python, along with standard libraries including [NumPy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/):

  1. a classical obstacle problem for the Poisson equation (Laplace operator) on the interval [0,1],
  2. the corresponding obstacle problem for the p-Laplacian operator, and
  3. the steady and implicit time-stepping obstacle problem for the shallow ice approximation (SIA) on an interval [0,L].

All of these are solved with a from-scratch implementation of the piecewise-linear (P1) finite element method.

## obstacle.py

Program `obstacle.py` solves 1D obstacle problems on an interval.  For detailed information do

        $ ./obstacle.py -h | less

The available solver algorithms are:

  * V-cycles and F-cycles (`-ni`) of the Tai (2003) multilevel constraint decomposition (MCD) method, both in a linear (MCDL) and a nonlinear FAS-type (MCDN) version, using the smoothers below

  * sweeps (`-sweepsonly`) of projected Gauss-Seidel (PGS) or Jacobi iteration smoothers, which require a Newton iteration in the nonlinear SIA case (PNGS, PNJacobi)

There are three obstacle problems:

  * `-problem poisson` solves the classical obstacle problem, with "ice-like" or "traditional" geometry

  * `-problem plap` solves a p=4 p-Laplacian problem, with "pile" or "bridge" geometry

  * `-problem sia` solves the steady-state shallow ice approximation for either a dome "profile" on a flat bed or on a bumpy bed

Illustration of the MCD method (V-cycles only) for the classical obstacle problem:

        $ ./obstacle.py -show -diagnostics -J 8

Illustration of the MCD method (F-cycle then V-cycles) for the SIA problem:

        $ ./obstacle.py -problem sia -show -diagnostics -J 8 -ni

Solve the classical obstacle problem slowly using single-level PGS:

        $ ./obstacle.py -sweepsonly -cyclemax 1000 -J 4

Higher resolution and different options for SIA:

        $ ./obstacle.py -problem sia -J 11 -jcoarse 1 -irtol 1.0e-7 -ni -nicycles 2 -show -diagnostics -monitor

## testing

Run software tests:

        $ make

First this calls `python3 -m pytest` to test modules.  (This should work whether or not `pytest` has been installed (e.g. via pip).)  Then it uses `../testit.sh` to compare results from `obstacle.py` runs to stored results in `output/`.

## dome/

This is old Firedrake code, superseded by [stokes-ice-tutorial](https://github.com/bueler/stokes-ice-tutorial)