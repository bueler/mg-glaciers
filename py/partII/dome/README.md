# mg-glaciers/py/1D/dome/

Generate and solve Stokes problems from Bueler profile geometry.

## COMMENT

This example is really superceded by the several examples in the following repo:

  * [github.com/bueler/stokes-ice-tutorial](https://github.com/bueler/stokes-ice-tutorial)

## Before solving

For the solve stages below one needs to activate the venv for firedrake:

        $ unset PETSC_DIR;  unset PETSC_ARCH;   # possibly needed
        $ source ~/firedrake/bin/activate

## Two-stage triangular mesh process

This case has two stages:
  * `domain.py` does the mesh-generation stage and saves the mesh
  * `solve.py` reads the mesh and solves the Stokes problem

For example, to generate a mesh of modest resolution do:

        $ ./domain.py -mx 60 -o dome60.geo
        $ gmsh -2 dome60.geo

View the mesh with `gmsh dome60.msh`.  Then solve:

        $ ./solve.py -mesh dome60.msh -s_snes_converged_reason -o dome60.pvd

Consider PETSc options `-s_snes_monitor`, `-s_snes_max_it`, and `-s_snes_rtol`
for controlling the Newton iteration.

View the result with `paraview dome60.pvd`.

## One-stage extruded quadrilateral mesh process

Generate the mesh and solve:

        $ ./solve.py -extrude -mx 60 -mz 5 -s_snes_converged_reason -o dome60ext.pvd

View the result with `paraview dome60ext.pvd`.

## Usage help

         $ ./domain.py -h
         $ ./solve.py -solvehelp        # options for solve.py
         $ ./solve.py -extrude -help    # PETSc options for solver
