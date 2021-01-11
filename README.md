# mg-glaciers

## paper/

This is a stub for a review paper I should write.

## py/

The code `obs1.py` in `py/` solves a 1D obstacle problem on [0,1] for the
Poisson equation.  For now the available algorithms are

  * sweeps of projected Gauss-Seidel (pGS) on the fine grid
  * V-cycles of the Tai (2003) multilevel subset decomposition method using pGS as a smoother

For detailed information:

        $ cd py/
        $ ./obs1.py -obs1help

A good illustration run using a classical obstacle problem geometry is:

        $ ./obs1.py -show -diagnostics -jfine 5 -jcoarse 2 -cycles 20 -random -fscale 10 -symmetric

An illustration of the ice-like problem geometry:

        $ ./obs1.py -problem icelike -show -diagnostics -jfine 4 -cycles 10

Run software tests:

        $ make test

## fas/

This is a paper and Python program which I wrote to make the full approximation storage (FAS) scheme blindingly clear to me.  See `fas/README` for details.

