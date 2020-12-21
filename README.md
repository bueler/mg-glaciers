# mg-glaciers

## paper/

This is a stub for a review paper I should write.

## py/

The code `obs1.py` in `py/` solves a 1D obstacle problem on [0,1] for the
Poisson equation.  For now the available algorithms are

  * sweeps of projected Gauss-Seidel (pGS) on the fine grid
  * V-cycles of a monotone multigrid method (Tai 2003) using pGS as a smoother

For detailed information:

        $ cd py/
        $ ./obs1.py -obs1help

A good illustration run is:

        $ ./obs1.py -show -diagnostics -jfine 5 -jcoarse 2 -cycles 20 -random -fvalue -20 -symmetric

Run software tests:

        $ make test

