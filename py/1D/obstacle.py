#!/usr/bin/env python3
'''Solve 1D obstacle problems by a multilevel constraint decomposition method.'''

import sys
import argparse
import numpy as np

from meshlevel import MeshLevel1D
from monitor import ObstacleMonitor
from visualize import VisObstacle

from smoothers.poisson import PsmootherPoisson
from smoothers.plap import PNsmootherPLap
from smoothers.sia import PNsmootherSIA

from mcdl import mcdlfcycle, mcdlsolver
from mcdn import mcdnfcycle, mcdnsolver

parser = argparse.ArgumentParser(description='''
Solve 1D obstacle problems:  For given Banach space X and obstacle phi in X,
find u in the closed, convex subset
    K = {v in X | v >= phi}
so that the variational inequality (VI) holds,
    F(u)[v-u] >= 0   for all v in K.
Note u solves an interior PDE in the inactive set {x | u(x) > phi(x)}.
We solve three such problems:

1. -problem poisson = classical obstacle problem:
    X = H_0^1[0,1]
    phi (obstacle)
    f (source) is in L^2[0,1]
    ell[v] = <f,v>
             /1
    a(u,v) = |  u'(x) v'(x) dx   (bilinear form)
             /0
    F(u)[v] = a(u,v) - ell[v]
    PDE is Poisson equation  - u'' = f

2. -problem plap = p-Laplacian (p=4) obstacle problem:
    X = W_0^{1,p}[0,1]
    phi (obstacle)
    f (source) is in L^q[0,1]
    ell[v] = <f,v>
              /1
    N(u)[v] = |  |u'|^{p-2} u' v dx
              /0
    F(u)[v] = N(u)[v] - ell[v]
    PDE is p-Laplacian equation  - (|u'|^{p-2} u')' = f

3. -problem sia = shallow ice approximation (SIA) obstacle problem:
    X = W_0^{1,p}[0,xmax]  where p = n + 1
    b = phi (bed elevation)
    a (mass balance) is in L^q[0,xmax]
    ell[v] = <a,v>
              /xmax
    N(s)[v] = |     Gamma (s-b)^{n+2} |s'|^{n-1} s' v dx
              /0
    F(s)[v] = N(s)[v] - ell[v]
    PDE is SIA equation  - (Gamma (s-b)^{n+2} |s'|^{n-1} s')' = a

Solution is by the multilevel constraint decomposition (MCD) method of
Tai (2003).  As in Alg. 4.7 of Gräser & Kornhuber (2009), we implement
MCD using a monotone restriction operator which decomposes the defect
obstacle.  Gräser & Kornhuber (2009) implement a down-slash cycle, but
we extend this to up-slash and v-cycles as well.  Our default is an
up-slash V(0,1) cycle.

The smoother and the coarse-mesh solver are either projected Gauss-Seidel
or projected Jacobi using a relaxation parameter (-omega).  These are
nonlinear for problem 2, using a fixed number of Newton iterations at each
point.  Option -sweepsonly reverts to using these smoothers on a single level.

Get usage help with -h.

References:
  * Bueler, E. (2016). Stable finite volume element schemes for the
    shallow ice approximation. J. Glaciol. 62, 230--242.
  * Bueler, E. (in prep.). Geometric multigrid for glacier modeling:
    New concepts and algorithms.
  * Gräser, C., & Kornhuber, R. (2009). Multigrid methods for
    obstacle problems. J. Comput. Math. 27 (1), 1--44.
  * Tai, X.-C. (2003). Rate of convergence for some constraint
    decomposition methods for nonlinear variational inequalities.
    Numer. Math. 93 (4), 755--786.
''',
    formatter_class=argparse.RawTextHelpFormatter,
    allow_abbrev=False)  # bug in python 3.8 causes this to be ignored
parser.add_argument('-coarse', type=int, default=1, metavar='N',
                    help='smoother sweeps on coarsest grid (default=1)')
parser.add_argument('-coarsestomega', type=float, default=1.0, metavar='X',
                    help='relaxation factor in smoother on coarsest level (default X=1.0)')
parser.add_argument('-cyclemax', type=int, default=100, metavar='N',
                    help='maximum number of (multilevel) cycles (default=100)')
parser.add_argument('-diagnostics', action='store_true', default=False,
                    help='generate residual and multilevel decomposition figures (use with -show or -o)')
parser.add_argument('-diagnosticsheuristic', action='store_true', default=False,
                    help='generate heuristic decomposition figure (use with -show or -o)')
parser.add_argument('-down', type=int, default=0, metavar='N',
                    help='smoother sweeps before coarse-mesh correction (default=0)')
parser.add_argument('-exactinitial', action='store_true', default=False,
                    help='initialize using exact solution')
parser.add_argument('-irtol', type=float, default=1.0e-3, metavar='X',
                    help='norm of inactive residual is reduced by this factor (default X=1.0e-3)')
parser.add_argument('-jacobi', action='store_true', default=False,
                    help='use Jacobi (additive) instead of Gauss-Seidel (multiplicative) for smoothing')
parser.add_argument('-jcoarse', type=int, default=0, metavar='J',
                    help='coarse mesh is jth level (default jcoarse=0 gives 1 node)')
parser.add_argument('-J', type=int, default=3, metavar='J',
                    help='fine mesh is Jth level (default J=3)')
parser.add_argument('-l1err', action='store_true', default=False,
                    help='report L^1 errors in addition to other norms')
parser.add_argument('-lsmonitor', action='store_true', default=False,
                    help='for PNGS, monitor added line-search point residual evaluations')
parser.add_argument('-mgview', action='store_true', default=False,
                    help='view multigrid cycles by indented print statements')
parser.add_argument('-monitor', action='store_true', default=False,
                    help='print the inactive-set residual norm after each cycle')
parser.add_argument('-monitorerr', action='store_true', default=False,
                    help='print the error (if available) after each cycle')
parser.add_argument('-newtonits', type=int, default=2, metavar='N',
                    help='Newton iterations in nonlinear smoothers (default N=1)')
parser.add_argument('-ni', action='store_true', default=False,
                    help='use nested iteration (F-cycle) for initial iterates')
parser.add_argument('-nicascadic', action='store_true', default=False,
                    help='scheduled nested iteration (implies -ni)')
parser.add_argument('-nicycles', type=int, default=1, metavar='N',
                    help='cycles in nested iteration before finest (default N=1)')
parser.add_argument('-nols', action='store_true', default=False,
                    help='for PNGS, do not use Armijo-based line search')
parser.add_argument('-o', metavar='FILE', type=str, default='',
                    help='save plot at end in image file, e.g. .pdf or .png')
parser.add_argument('-omega', type=float, default=1.0, metavar='X',
                    help='relaxation factor in smoother (default X=1.0)')
parser.add_argument('-plain', action='store_true', default=False,
                    help='when used with -show or -o, only show exact solution and obstacle')
parser.add_argument('-plapcase', choices=['pile','bridge'], metavar='X',
                    default='pile',
                    help='determines obstacle and source function (default: %(default)s)')
parser.add_argument('-poissoncase',
                    choices=['icelike', 'traditional', 'pde1', 'pde2'],
                    metavar='X', default='icelike',
                    help='determines obstacle and source function (default: %(default)s)')
parser.add_argument('-poissonfscale', type=float, default=1.0, metavar='X',
                    help='in Poisson equation -u"=f this multiplies f (default X=1.0)')
parser.add_argument('-poissonparabolay', type=float, default=-1.0, metavar='X',
                    help='vertical location of obstacle (default X=-1.0)')
parser.add_argument('-printwarnings', action='store_true', default=False,
                    help='print pointwise feasibility warnings')
parser.add_argument('-problem', choices=['poisson', 'plap', 'sia'], metavar='X', default='poisson',
                    help='determines obstacle problem (default: %(default)s)')
parser.add_argument('-random', action='store_true', default=False,
                    help='make a smooth random perturbation of the obstacle')
parser.add_argument('-randomscale', type=float, default=1.0, metavar='X',
                    help='scaling of modes in -random perturbation (default X=1.0)')
parser.add_argument('-randomseed', type=int, default=1, metavar='X',
                    help='seed the generator in -random perturbation (default X=1)')
parser.add_argument('-randommodes', type=int, default=30, metavar='N',
                    help='number of sinusoid modes in -random perturbation (default N=30)')
parser.add_argument('-show', action='store_true', default=False,
                    help='show plot at end')
parser.add_argument('-showsingular', action='store_true', default=False,
                    help='on each sweep on each level, show where the point Jacobian was singular')
parser.add_argument('-siacase', choices=['profile', 'bumpy'],
                    metavar='X', default='profile',
                    help='determines bed elevation (default: %(default)s)')
parser.add_argument('-siacupmax', type=float, default=100.0, metavar='X',
                    help='in SIA, do not move surface up by more than this (default X=100.0 m)')
parser.add_argument('-siaeps', type=float, default=0.0, metavar='X',
                    help='regularizing thickness in SIA smoother (default 0 m)')
parser.add_argument('-siaintervallength', type=float, default=1800.0e3,
                    metavar='L',
                    help='solve SIA on [0,L] (default L=1800 km)')
parser.add_argument('-sweepsnotalternate', action='store_true', default=False,
                    help='normally sweeps alternate direction; stop doing this')
parser.add_argument('-sweepsonly', action='store_true', default=False,
                    help='do smoother sweeps as cycles, instead of multilevel')
parser.add_argument('-up', type=int, default=2, metavar='N',
                    help='smoother sweeps after coarse-mesh correction (default=2)')
args, unknown = parser.parse_known_args()

# provide usage help
if len(unknown) > 0:
    print('usage ERROR: unknown arguments ... try -h or --help for usage')
    sys.exit(1)
if args.show and args.o:
    print('usage ERROR: use either -show or -o FILE but not both')
    sys.exit(2)
if args.down < 0 or args.coarse < 0 or args.up < 0:
    print('usage ERROR: -down, -coarse, -up must be nonnegative integers')
    sys.exit(3)
if args.down + args.up == 0:
    print('WARNING: You have set -down 0 and -up 0.  Not convergent without smoothing.')
if args.nicascadic:
    args.ni = True
if args.ni and args.random:
    raise NotImplementedError('combination of -ni and -random is not implemented')

# determine correct interval
L = 1.0
if args.problem == 'poisson' and args.poissoncase == 'pde2':
    L = 10.0
elif args.problem == 'sia':
    L = args.siaintervallength

# hierarchy is a list of MeshLevel1D with indices [0,..,levels-1]
assert args.jcoarse >= 0
assert args.J >= args.jcoarse
levels = args.J - args.jcoarse + 1
hierarchy = [None] * (levels)             # list [None,...,None]
for j in range(levels):
    hierarchy[j] = MeshLevel1D(j=j+args.jcoarse, xmax=L)

# set up obstacle problem with smoother (class SmootherObstacleProblem)
if args.problem == 'poisson':
    obsprob = PsmootherPoisson(args)
elif args.problem == 'plap':
    obsprob = PNsmootherPLap(args)
elif args.problem == 'sia':
    obsprob = PNsmootherSIA(args)

# more usage help
if args.monitorerr and not obsprob.exact_available():
    print('usage ERROR: -monitorerr but exact solution and error not available')
    sys.exit(5)

# attach fields to hierarchy if needed
if args.problem in ['plap', 'sia']:
    # attach interpolated bed elevation to each mesh level
    for j in range(levels):
        hierarchy[j].b = obsprob.phi(hierarchy[j].xx())
    if args.sweepsonly:
        # nonlinear cases: fixed part of iterate is zero in sweepsonly
        for j in range(levels):
            hierarchy[j].g = hierarchy[j].zeros()

# fine-level problem data
mesh = hierarchy[-1]
phi = obsprob.phi(mesh.xx())
ellf = mesh.ellf(obsprob.source(mesh.xx()))  # source functional ell[v] = <f,v>

# create fine-level monitor (using exact solution if available)
uex = None
if obsprob.exact_available():
    uex = obsprob.exact(mesh.xx())
mon = ObstacleMonitor(obsprob, mesh, uex=uex,
                      printresiduals=args.monitor, printerrors=args.monitorerr,
                      l1err=args.l1err,
                      extraerrorpower=obsprob.rr if args.problem == 'sia' else None,
                      extraerrornorm=obsprob.pp if args.problem == 'sia' else None)

# initialization on fine mesh
if args.exactinitial:
    uu = obsprob.exact(mesh.xx())
else:
    uu = obsprob.initial(mesh.xx())

# residual norm for initial iterate
irnorm, _ = mon.irerr(uu, ellf, phi, indent=0)

# decide on using MCDL or MCDN
linear = (args.problem == 'poisson')

# do F-cycle first if requested; counts as first iterate
if args.ni:
    if linear:
        uu = mcdlfcycle(args, obsprob, levels-1, hierarchy)
    else:
        uu = mcdnfcycle(args, obsprob, levels-1, hierarchy)

# apply sweeps of fine-level smoother (versus mcdXsolver())
def sweepssolver(args, obsprob, mesh, ellf, phi, w, monitor,
                 iters=100, irnorm0=None):
    if irnorm0 == 0.0:
        return
    for s in range(iters):
        # smoother sweeps on finest level
        obsprob.smoothersweep(mesh, w, ellf, phi)
        irnorm, errnorm = monitor.irerr(w, ellf, phi, indent=0)
        if irnorm > 100.0 * irnorm0:
            print('WARNING:  irnorm > 100 irnorm0')
        if irnorm <= args.irtol * irnorm0:
            break

# solve to tolerance
itermax = args.cyclemax
if args.ni:
    itermax -= args.nicycles
if itermax > 0:
    if args.sweepsonly:
        sweepssolver(args, obsprob, hierarchy[-1], ellf, phi, uu, mon,
                     iters=itermax, irnorm0=irnorm)
    else:
        if linear:
            mcdlsolver(args, obsprob, levels-1, hierarchy, ellf, phi, uu, mon,
                       iters=itermax, irnorm0=irnorm)
        else:
            mcdnsolver(args, obsprob, levels-1, hierarchy, ellf, uu, mon,
                       iters=itermax, irnorm0=irnorm)

# accumulate work units from values stored in hierarchy
WUsum = 0.0
for j in range(levels):
    WUsum += hierarchy[j].WU / 2**(levels - 1 - j)

# report on computation including numerical error, WU, infeasibles
method = ''
if args.ni:
    method = 'F-cycle + '
if args.sweepsonly:
    method += '%d applications of %s smoother' % (mon.s - 1, obsprob.name)
else:
    method += '%d %s V(%d,%d) cycles' % (mon.s - 1, obsprob.name, args.down, args.up)
if obsprob.exact_available():
    uex = obsprob.exact(hierarchy[-1].xx())
    if args.l1err:
        error = ';  |u-uexact|_1 = %.3e, |u-uexact|_2 = %.3e' \
                % (mesh.l1norm(uu - uex), mesh.l2norm(uu - uex))
    else:
        error = ';  |u-uexact|_2 = %.3e' % mesh.l2norm(uu - uex)
    if args.problem == 'sia':
        error += ', |u^r-uexact^r|_p = %.3e' % \
                 mesh.lqnorm(obsprob.pp, uu**obsprob.rr - uex**obsprob.rr)
    elif args.problem == 'plap':
        error += ', |u-uexact|_p = %.3e' % mesh.lqnorm(obsprob.pp, uu - uex)
else:
    uex = None
    error = ''
inadstr = ''
if obsprob.inadmissible > 0:
    inadstr = ' (%d inadmissibles)' % obsprob.inadmissible
print('J=%d (m=%d): %s -> %.3f WU%s%s' \
      % (args.J, mesh.m, method, WUsum, error, inadstr))

# graphical output if desired
if args.show or args.o or args.diagnostics:
    vis = VisObstacle(args, obsprob, hierarchy,
                      u=uu, phi=phi, ell=ellf, uex=uex)
    vis.generate()
