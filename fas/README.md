# mg-glaciers/fas/

The program `fas1.py` in directory `py/` demonstrates by example how the _full approximation storage_ (FAS) multigrid scheme works.  It solves a nonlinear ODE BVP using piecewise-linear finite elements and a nonlinear Gauss-Seidel smoother.

## Documentation

Read `fas.pdf` in `doc/` after generating it:

        $ cd doc/
        $ make

Clean up the LaTeX clutter in `doc/` with

        $ make clean

## Run the program

To get started do

        $ cd py/

Next do one thing which is helpful to bypass any Python package and/or relative import nonsense.  Namely, add a symbolic link to a needed (but very simple) module in `mg-glaciers/py/`:

        $ ln -s ../../py/meshlevel.py

Now you should be able to do the following simple run using three FAS V-cycles on a mesh of 64 subintervals:

        $ ./fas1.py -j 6 -cycles 3 -monitor -show

A convergence study uses a version of the problem with a known exact solution (method of manufactured solutions):

        $ for J in 1 2 3 4 5 6 7; do ./fas1.py -j $J -cycles 5 -mms; done

FIXME demo optimality

Run software tests:

        $ make test

