#!/usr/bin/env python3
'''Solve steady-geometry Stokes obstacle problem by a multilevel constraint decomposition method.'''

# TODO:
#   1. improve checkpointing into .pvd to include stresses and effective
#      viscosity (see stokes-ice-tutorial stage 4,5)
#   2. write some basic run tests with ../testit.sh (see partI/makefile)
#   3. use SIA time-step criterion to set alpha in NRich
#   4. implement smoothersweep() based on NRich
#   5. copy partI/mcdn.py and build it out

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
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
    allow_abbrev=False,  # bug in python 3.8 causes this to be ignored
    add_help=False)
adda = parser.add_argument
adda('-coarse', type=int, default=1, metavar='N',
     help='smoother sweeps on coarsest grid (default=1)')
adda('-cyclemax', type=int, default=100, metavar='N',
     help='maximum number of (multilevel) cycles (default=100)')
adda('-domainlength', type=float, default=30.0e3, metavar='L',
     help='solve on [0,L] (default L=30 km)')
adda('-down', type=int, default=0, metavar='N',
     help='smoother sweeps before coarse-mesh correction (default=0)')
adda('-eps', type=float, metavar='X', default=1.0e-2,  # FIXME sensitive
    help='regularization used in viscosity (default=10^{-2})')
adda('-Hmin', type=float, metavar='X', default=0.0,
    help='minimum ice thickness; Hmin>0 for cliffs or padding (default=0.0)')
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
adda('-padding', action='store_true', default=False,
     help='put Hmin thickness of ice in ice-free locations')
adda('-printwarnings', action='store_true', default=False,
     help='print pointwise feasibility warnings')
adda('-steadyhelp', action='store_true', default=False,
     help='print help for steady.py and end (vs -help for PETSc options)')
adda('-sweepsonly', action='store_true', default=False,
     help='do smoother sweeps as cycles, instead of multilevel')
adda('-up', type=int, default=2, metavar='N',
     help='smoother sweeps after coarse-mesh correction (default=2)')
args, unknown = parser.parse_known_args()
if args.steadyhelp:
    parser.print_help()
    sys.exit(0)

# mesh hierarchy: a list of MeshLevel1D with indices [0,..,levels-1]
assert args.J >= args.jcoarse >= 0
levels = args.J - args.jcoarse + 1
hierarchy = [None] * (levels)             # list [None,...,None]
for j in range(levels):
    hierarchy[j] = MeshLevel1D(j=j+args.jcoarse, xmax=args.domainlength)

obsprob = SmootherStokes(args)

# fine-level problem data
mesh = hierarchy[-1]
phi = obsprob.phi(mesh.xx())
ellf = mesh.ellf(obsprob.source(mesh.xx()))  # source functional ell[v] = <f,v>
s = obsprob.initial(mesh.xx())
#obsprob.shownonzeros(s)

b = mesh.zeros()

def inactiveresidualnorm(s, r, ireps=50.0):
    '''Compute the norm of the residual values at nodes where the constraint
    is NOT active.  Note that where the constraint is active the residual F(s)
    in the complementarity problem is allowed to have any positive value, and
    only the residual at inactive nodes is relevant to convergence.'''
    F = r.copy()
    F[s < b + ireps] = np.minimum(F[s < b + ireps], 0.0)
    return mesh.l2norm(F)

def output(filename, description):
    '''Either save result to an image file or use show().  Supply '' as filename
    to use show().'''
    if len(filename) == 0:
        plt.show()
    else:
        print('saving %s to %s ...' % (description, filename))
        plt.savefig(filename, bbox_inches='tight')

def final(mesh, s, cmb, filename=''):
    '''Generate graphic showing final iterate and CMB function.'''
    secpera = 31556926.0
    mesh.checklen(s)
    xx = mesh.xx()
    xx /= 1000.0
    plt.figure(figsize=(15.0, 8.0))
    plt.subplot(2,1,1)
    plt.plot(xx, s, 'k', linewidth=4.0)
    plt.xlabel('x (km)')
    plt.ylabel('surface elevation (m)')
    plt.subplot(2,1,2)
    plt.plot(xx, cmb * secpera, 'r')
    plt.grid()
    plt.ylabel('CMB (m/a)')
    plt.xlabel('x (km)')
    output(filename, 'final iterate and obstacle')

# FIXME be more sophisticated ... but the following sort of works

# simple loop to do projected nonlinear Richardson, = explicit time-stepping,
# as a candidate smoother
rtol = 1.0e-4
alpha = 1000.0  # good for J=3,4,5 ... maybe
s0 = s.copy()
r = obsprob.residual(mesh, s, ellf, savename='step0.pvd')
normF0 = inactiveresidualnorm(s, r)
print('0: %.4e' % normF0)
for j in range(args.cyclemax):
    s = np.maximum(s - alpha * r, 0.0)
    r = obsprob.residual(mesh, s, ellf)
    normF = inactiveresidualnorm(s, r)
    print('%d: %.4e' % (j+1, normF))
    if normF < rtol * normF0:
        break
r = obsprob.residual(mesh, s, ellf, savename='step%d.pvd' % (j+1))

final(mesh, s, obsprob.source(mesh.xx()))
# FIXME much more to do
