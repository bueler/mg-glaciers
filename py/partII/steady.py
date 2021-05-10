#!/usr/bin/env python3
'''Solve steady-geometry Stokes obstacle problem by a multilevel constraint decomposition method.'''

import sys
import argparse
import numpy as np
from firedrake import *

from meshlevel import MeshLevel1D
from smoother import SmootherStokes
#from mcdn import mcdnsolver

parser = argparse.ArgumentParser(description='''
Solve the 1D obstacle problem for the steady geometry of a glacier using
Stokes dynamics:  For given climate a(x) and bed elevation b(x),
find s(x) in the closed, convex subset
    K = {r | r >= b}
so that the variational inequality (VI) holds,
    F(s)[r-s] >= 0   for all r in K.
Note s solves the surface kinematical equation, as an interior PDE, in the
inactive set {x | s(x) > b(x)}.

Solution is by the nonlinear (FAS) extension of the multilevel constraint
decomposition (MCD) method of Tai (2003).  See the simpler cases, including
the SIA, in part I (Bueler, 2022).

Initial implementation generates Bueler profile geometry as initial state
and then tries to converge from there.

References:
  * Bueler, E. (2022). Geometric multigrid for glacier modeling I:
    New concepts and algorithms.  In preparation.
  * Tai, X.-C. (2003). Rate of convergence for some constraint
    decomposition methods for nonlinear variational inequalities.
    Numer. Math. 93 (4), 755--786.
''',
    formatter_class=argparse.RawTextHelpFormatter,
    allow_abbrev=False)  # bug in python 3.8 causes this to be ignored
adda = parser.add_argument
adda('-coarse', type=int, default=1, metavar='N',
     help='smoother sweeps on coarsest grid (default=1)')
adda('-cyclemax', type=int, default=100, metavar='N',
     help='maximum number of (multilevel) cycles (default=100)')
adda('-domainlength', type=float, default=30.0e3, metavar='L',
     help='solve on [0,L] (default L=30 km)')
adda('-down', type=int, default=0, metavar='N',
     help='smoother sweeps before coarse-mesh correction (default=0)')
adda('-eps', type=float, metavar='X', default=1.0e-2,
    help='regularization used in viscosity (default=10^{-2})')
adda('-irtol', type=float, default=1.0e-3, metavar='X',
     help='reduce norm of inactive residual (default X=1.0e-3)')
adda('-jcoarse', type=int, default=0, metavar='J',
     help='coarse mesh is jth level (default jcoarse=0 gives 1 node)')
adda('-J', type=int, default=3, metavar='J',
     help='fine mesh is Jth level (default J=3)')
adda('-mz', type=int, default=4, metavar='MZ',
     help='number of (x,z) extruded mesh levels (default MZ=4)')
adda('-newtonits', type=int, default=2, metavar='N',
     help='Newton iterations in nonlinear smoothers (default N=1)')
adda('-o', metavar='FILE', type=str, default='',
     help='save plot at end in image file, e.g. .pdf or .png')
adda('-omega', type=float, default=1.0, metavar='X',
     help='relaxation factor in smoother (default X=1.0)')
adda('-printwarnings', action='store_true', default=False,
     help='print pointwise feasibility warnings')
adda('-sweepsonly', action='store_true', default=False,
     help='do smoother sweeps as cycles, instead of multilevel')
adda('-up', type=int, default=2, metavar='N',
     help='smoother sweeps after coarse-mesh correction (default=2)')
args, unknown = parser.parse_known_args()

# mesh hierarchy: a list of MeshLevel1D with indices [0,..,levels-1]
assert args.jcoarse >= 0
assert args.J >= args.jcoarse
L = args.domainlength
levels = args.J - args.jcoarse + 1
hierarchy = [None] * (levels)             # list [None,...,None]
for j in range(levels):
    hierarchy[j] = MeshLevel1D(j=j+args.jcoarse, xmax=L)

obsprob = SmootherStokes(args)

# fine-level problem data
mesh = hierarchy[-1]
phi = obsprob.phi(mesh.xx())
ellf = mesh.ellf(obsprob.source(mesh.xx()))  # source functional ell[v] = <f,v>
s = obsprob.initial(mesh.xx())
#obsprob.showsingular(s)

# solve Stokes on the domain and compute residual of surface kinematical equation
obsprob.residual(mesh, s, ellf)
