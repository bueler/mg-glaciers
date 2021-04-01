# mg-glaciers/py/1D

Program `obstacle.py` solves 1D obstacle problems on an interval.  For detailed information do

        $ ./obstacle.py -h | less

The available solver algorithms are:

  * V-cycles and F-cycles (`-ni`) of the Tai (2003) multilevel constraint decomposition (MCD) method, both in a linear (MCDL) and a nonlinear FAS-type (MCDN) version, using the smoothers below

  * sweeps (`-sweepsonly`) of projected Gauss-Seidel (PGS) or Jacobi iteration smoothers, which require a Newton iteration in the nonlinear SIA case (PNGS, PNJacobi)

There are three obstacle problems:

  * `-problem poisson` solves the classical obstacle problem, with default "ice-like" geometry

  * `-problem plap` solves a p=4 p-Laplacian problem, with "bridge" geometry

  * `-problem sia` solves the shallow ice approximation for a steady-state dome on a flat bed

Illustration of the MCD method (V-cycles only) for the classical obstacle problem:

        $ ./obstacle.py -show -diagnostics -J 8

Illustration of the MCD method (F-cycle then V-cycles) for the SIA problem:

        $ ./obstacle.py -problem sia -show -diagnostics -J 8 -ni -symmetric

Solve the classical obstacle problem slowly using single-level PGS:

        $ ./obstacle.py -sweepsonly -cyclemax 1000 -J 4

Higher resolution and different options for SIA:

        $ ./obstacle.py -problem sia -J 11 -jcoarse 1 -irtol 1.0e-7 -ni -nicycles 2 -show -diagnostics -monitor

## testing

Run software tests:

        $ make

First this calls `python3 -m pytest` to test modules.  (This should work whether or not `pytest` has been installed (e.g. via pip).)  Then it uses `../testit.sh` to compare results from `obstacle.py` runs to stored results in `output/`.
