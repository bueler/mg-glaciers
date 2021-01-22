# mg-glaciers/py/

Program `obstacle.py` in `py/` solves a 1D obstacle problem on [0,1] for the Poisson equation, with "ice-like" geometry.  For now the available algorithms are

  * sweeps of projected Gauss-Seidel (pGS) on the fine grid
  * V-cycles of the Tai (2003) multilevel subset decomposition method using pGS as a smoother

For detailed information:

        $ ./obstacle.py -h

An illustration of the default ice-like problem geometry:

        $ ./obstacle.py -show -diagnostics -kfine 4 -cycles 10

A good illustration using classical obstacle problem geometry, and some different options, is:

        $ ./obstacle.py -problem parabola -show -diagnostics -kfine 5 -kcoarse 2 -cycles 20 -random -fscale 10 -symmetric


## testing

Run software tests:

        $ make test

This calls `python3 -m pytest`, which should work whether or not `pytest` has been installed (e.g. via pip).

