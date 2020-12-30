# mg-glaciers/fas/py/

The code `fas1.py` demonstrates by example how the _full approximation storage_ (FAS) multigrid scheme works.  It solves a nonlinear ODE BVP using piecewise-linear finite elements and a nonlinear Gauss-Seidel smoother.

To get started you should do one thing which is helpful to bypass any Python package and/or relative import nonsense.  Namely, add a symbolic link to a needed (but very simple) module in `mg-glaciers/py/`:

        $ ln -s ../../py/meshlevel.py

Now you should be able to do the following simple run using three FAS V-cycles on a mesh of 64 subintervals:

        $ ./fas1.py -j 6 -cycles 3 -monitor -show

A convergence study uses a version of the problem with a known exact solution (method of manufactured solutions):

        $ for J in 1 2 3 4 5 6 7; do ./fas1.py -j $J -cycles 5 -mms; done

Run software tests:

        $ make test

