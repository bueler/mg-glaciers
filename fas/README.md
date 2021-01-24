# mg-glaciers/fas/

This little project is finished.  I posted the documentation as a preprint to [arxiv.org/abs/2101.05408](https://arxiv.org/abs/2101.05408).  I do not plan to pursue publication.

The program `fas1.py` in directory `py/` demonstrates by example how the _full approximation storage_ (FAS) multigrid scheme works.  It solves an easy nonlinear ODE BVP using piecewise-linear finite elements and a nonlinear Gauss-Seidel smoother.

## Documentation (the preprint)

Read `fas.pdf` in `doc/` after generating it:

        $ cd doc/
        $ make

Clean up the LaTeX clutter in `doc/` with

        $ make clean

## Run the program

To get started do

        $ cd py/
        $ ./fas1.py -h

Do the following simple run using three FAS V-cycles on a mesh of 2^6=64 subintervals:

        $ ./fas1.py -K 6 -cycles 3 -monitor -show

See the convergence study, using a version of the problem with a known exact solution (method of manufactured solutions), in `py/study/converge.py`.  Also see the solver complexity (optimality) study of runtime and work units in `py/study/optimal.py` and `py/study/slow.py`.  These are described in section 6 of `doc/fas.pdf`.

Run software tests:

        $ make test

