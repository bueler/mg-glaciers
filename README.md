# mg-glaciers

## paper/

This is a stub for a review paper I should write.

## py/

The code `obstacle1.py` solves a 1D obstacle problem on [0,1] for the
Poisson equation.  For now the available algorithms are

  * sweeps of projected Gauss-Seidel (pGS) on the fine grid
  * V-cycles of the Tai (2003) monotone multigrid method

For detailed information:

        $ ./obstacle1.py -obstacle1help

Run software tests:

        $ make test

