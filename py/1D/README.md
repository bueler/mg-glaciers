# mg-glaciers/py/1D

Program `obstacle.py` solves a 1D obstacle problem on [0,1] for the Poisson equation, i.e. a classical obstacle problem, with default "ice-like" geometry.  For now the available solver algorithms are

  * sweeps of projected Gauss-Seidel (pGS) on the fine grid
  * slash-cycles of the Tai (2003) multilevel subset decomposition method using pGS as a smoother

For detailed information:

        $ ./obstacle.py -h

An illustration of the default ice-like problem geometry:

        $ ./obstacle.py -show -diagnostics -jfine 4

A good illustration using traditional obstacle problem geometry, and some different options, is:

        $ ./obstacle.py -problem parabola -show -diagnostics -jfine 5 -jcoarse 2 -random -fscale 10 -symmetric

## testing

Run software tests:

        $ make test

First this calls `python3 -m pytest` to test modules.  (This should work whether or not `pytest` has been installed (e.g. via pip).)  Then it uses `../testit.sh` to compare to stored results in `output/`.
