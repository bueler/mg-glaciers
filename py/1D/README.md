# mg-glaciers/py/1D

Program `obstacle.py` solves 1D obstacle problems on an interval.  Option `-problem poisson` solves the classical obstacle problem, with default "ice-like" geometry.  Option `-problem sia` solves the shallow ice approximation for a steady-state dome on a flat bed.

The available solver algorithms are

  * sweeps of projected Gauss-Seidel (PGS) or Jacobi iterations, with a Newton iteration in the SIA case
  * slash-cycles of the Tai (2003) multilevel constraint decomposition (MCD) method using the above sweeps as a smoother

For detailed information:

        $ ./obstacle.py -h | less

Illustration of the MCD method for the classical obstacle problem:

        $ ./obstacle.py -show -diagnostics -jfine 4

Solve the same problem slowly using single-level PGS:

        $ ./obstacle.py -sweepsonly -cyclemax 1000 -jfine 4

Higher resolution and different options:

        $ ./obstacle.py -jfine 10 -jcoarse 1 -irtol 1.0e-7 -random -fscale 2 -ni -show -diagnostics -monitor

## testing

Run software tests:

        $ make test

First this calls `python3 -m pytest` to test modules.  (This should work whether or not `pytest` has been installed (e.g. via pip).)  Then it uses `../testit.sh` to compare to stored results in `output/`.
