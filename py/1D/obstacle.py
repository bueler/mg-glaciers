#!/usr/bin/env python3
'''Solve a 1D obstacle problem by a multilevel constraint decomposition method.'''

# best observed settings for generating solutions in 10 WU in exact solutions cases:
#   for PROB in icelike parabola; do
#     for JJ in 6 7 8 9 10 11 12 13 14 15 16 17; do
#       ./obstacle.py -problem $PROB -jfine $JJ -ni -nicycles 2 -cyclemax 3 -omega 1.5
#     done
#   done
# but in -random cases it seems better to use -omega 1.0 and allow more cycles

import sys
import argparse
import numpy as np

from meshlevel import MeshLevel1D
from subsetdecomp import mcdlcycle
from monitor import ObstacleMonitor
from visualize import VisObstacle

from smoother import PGSPoisson, PJacobiPoisson
from siasmoother import PNGSSIA, PNJacobiSIA

parser = argparse.ArgumentParser(description='''
Solve 1D obstacle problems:  For given Banach space X and phi in X,
find u in the closed, convex subset
    K = {v in X | v >= phi}
so that the variational inequality holds,
    F(u)[v-u] >= ell[v-u]   for all v in K.
Where the constraint is inactive (u(x) > phi(x)), u solves an
interior PDE.  We solve two particular problems:

1. For classical obstacle problem in smoother.py:
    X = H_0^1[0,1]
    phi (obstacle) is in X
    f (source) is in L^2[0,1] (or X')
    ell[v] = <f,v>
                       /1
    F(u)[v] = a(u,v) = |  u'(x) v'(x) dx   (bilinear form)
                       /0
    interior PDE: Poisson equation  - u'' = f

2. For shallow ice approximation (SIA) obstacle problem (Bueler 2016) in
   siasmoother.py:
    X = W_0^{1,p}[0,xmax]  where p = n + 1
    b = phi (bed elevation) is in X
    m (mass balance) is in L^2[0,xmax] (or X')
    ell[v] = <m,v>
              /xmax
    F(s)[v] = |     Gamma (s-b)^{n+2} |s'|^{n-1} s' v dx
              /0
    interior PDE: SIA equation  - (Gamma (s-b)^{n+2} |s'|^{n-1} s')' = m

Solution is by Alg. 4.7 in Gräser & Kornhuber (2009), namely the multilevel
constraint decomposition slash-cycle method by Tai (2003) in which a monotone
restriction operator decomposes the defect obstacle.  The smoother and the
coarse-mesh solver are either projected Gauss-Seidel or projected Jacobi
using a relaxation parameter (-omega).  These are nonlinear for problem 2,
using a fixed number of Newton iterations at each point.  Option -sweepsonly
reverts to a single-level solver using these smoothers.

Get usage help with -h or --help.

References:
  * Bueler, E. (2016). Stable finite volume element schemes for the
    shallow ice approximation. J. Glaciol. 62, 230--242.
  * Gräser, C., & Kornhuber, R. (2009). Multigrid methods for
    obstacle problems. J. Comput. Math. 27 (1), 1--44.
  * Tai, X.-C. (2003). Rate of convergence for some constraint
    decomposition methods for nonlinear variational inequalities.
    Numer. Math. 93 (4), 755--786.
''', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-coarse', type=int, default=1, metavar='N',
                    help='PGS sweeps on coarsest grid (default=1)')
parser.add_argument('-coarsestomega', type=float, default=1.0, metavar='X',
                    help='relaxation factor in PGS, thus PSOR, on coarsest level (default X=1.0)')
parser.add_argument('-cyclemax', type=int, default=100, metavar='N',
                    help='maximum number of multilevel cycles (default=100)')
parser.add_argument('-diagnostics', action='store_true', default=False,
                    help='additional diagnostics figures (use with -show or -o)')
parser.add_argument('-down', type=int, default=1, metavar='N',
                    help='PGS sweeps before coarse-mesh correction (default=1)')
parser.add_argument('-errtol', type=float, default=None, metavar='X',
                    help='stop if numerical error (if available) below X (default=None)')
parser.add_argument('-exactinitial', action='store_true', default=False,
                    help='initialize using exact solution')
parser.add_argument('-fscale', type=float, default=1.0, metavar='X',
                    help='in Poisson equation -u"=f this multiplies f (default X=1.0)')
parser.add_argument('-irtol', type=float, default=1.0e-3, metavar='X',
                    help='norm of inactive residual is reduced by this factor (default X=1.0e-3)')
parser.add_argument('-jacobi', action='store_true', default=False,
                    help='use Jacobi (additive) instead of Gauss-Seidel (multiplicative) for smoothing')
parser.add_argument('-jcoarse', type=int, default=0, metavar='J',
                    help='coarse mesh is jth level (default jcoarse=0 gives 1 node)')
parser.add_argument('-jfine', type=int, default=3, metavar='J',
                    help='fine mesh is jth level (default jfine=3)')
parser.add_argument('-mgview', action='store_true', default=False,
                    help='view multigrid cycles by indented print statements')
parser.add_argument('-monitor', action='store_true', default=False,
                    help='print the inactive-set residual norm after each cycle')
parser.add_argument('-monitorerr', action='store_true', default=False,
                    help='print the error (if available) after each cycle')
parser.add_argument('-ni', action='store_true', default=False,
                    help='use nested iteration for initial iterates (= F-cycle)')
parser.add_argument('-nicascadic', action='store_true', default=False,
                    help='scheduled nested iteration (implies -ni)')
parser.add_argument('-nicycles', type=int, default=1, metavar='N',
                    help='nested iteration: cycles on levels before finest (default N=1)')
parser.add_argument('-o', metavar='FILE', type=str, default='',
                    help='save plot at end in image file, e.g. PDF or PNG')
parser.add_argument('-omega', type=float, default=1.0, metavar='X',
                    help='relaxation factor in PGS, thus PSOR (default X=1.0)')
parser.add_argument('-parabolay', type=float, default=-1.0, metavar='X',
                    help='vertical location of obstacle in -problem parabola (default X=-1.0)')
parser.add_argument('-plain', action='store_true', default=False,
                    help='when used with -show or -o, only show exact solution and obstacle')
parser.add_argument('-poissoncase', choices=['icelike', 'parabola'], metavar='X', default='icelike',
                    help='determines obstacle and source function (default: %(default)s)')
parser.add_argument('-printwarnings', action='store_true', default=False,
                    help='print pointwise feasibility warnings')
parser.add_argument('-problem', choices=['poisson', 'sia'], metavar='X', default='poisson',
                    help='determines entire obstacle problem (default: %(default)s)')
parser.add_argument('-random', action='store_true', default=False,
                    help='make a smooth random perturbation the obstacle')
parser.add_argument('-randomscale', type=float, default=1.0, metavar='X',
                    help='scaling of modes in -random perturbation (default X=1.0)')
parser.add_argument('-randomseed', type=int, default=1, metavar='X',
                    help='seed the generator in -random perturbation (default X=1)')
parser.add_argument('-randommodes', type=int, default=30, metavar='N',
                    help='number of sinusoid modes in -random perturbation (default N=30)')
parser.add_argument('-show', action='store_true', default=False,
                    help='show plot at end')
parser.add_argument('-siaintervallength', type=float, default=1800.0e3, metavar='L',
                    help='solve SIA on [0,L] (default L=1800 km)')
parser.add_argument('-sweepsonly', action='store_true', default=False,
                    help='do smoother sweeps (PGS or PJacobi) as cycles, instead of multilevel')
parser.add_argument('-symmetric', action='store_true', default=False,
                    help='use symmetric projected Gauss-Seidel sweeps (forward then backward)')
parser.add_argument('-up', type=int, default=0, metavar='N',
                    help='PGS sweeps after coarse-mesh correction (default=0; up>0 is V-cycle)')
args, unknown = parser.parse_known_args()

# provide usage help
if unknown:
    print('usage ERROR: unknown arguments ... try -h or --help for usage')
    sys.exit(1)
if args.show and args.o:
    print('usage ERROR: use either -show or -o FILE but not both')
    sys.exit(2)
if args.nicascadic:
    args.ni = True

# hierarchy will be a list of MeshLevel1D with indices [0,..,levels-1]
assert args.jcoarse >= 0
assert args.jfine >= args.jcoarse
levels = args.jfine - args.jcoarse + 1
hierarchy = [None] * (levels)             # list [None,...,None]

# set up obstacle problem with smoother (class SmootherObstacleProblem)
#   and meshes on correct interval
if args.problem == 'poisson':
    if args.jacobi:
        obsprob = PJacobiPoisson(args)
    else:
        obsprob = PGSPoisson(args)
    for j in range(levels):
        hierarchy[j] = MeshLevel1D(j=j+args.jcoarse, xmax=1.0)
elif args.problem == 'sia':
    if args.jacobi:
       obsprob = PNGSSIA(args)
    else:
       obsprob = PNGSSIA(args)
    for j in range(levels):
        hierarchy[j] = MeshLevel1D(j=j+args.jcoarse,
                                   xmax=args.siaintervallength)
    # attach obstacle to mesh
    # FIXME o.k for single level but NOT for multilevel
    if not args.sweepsonly:
        raise NotImplementedError( \
            'The constraint decomposition theory is not ready for SIA.  Use -sweepsonly.')
    mesh = hierarchy[-1]
    mesh.phi = obsprob.phi(mesh.xx())

# more usage help
if args.monitorerr and not obsprob.exact_available():
    print('usage ERROR: -monitorerr but exact solution and error not available')
    sys.exit(3)
if args.errtol is not None and not obsprob.exact_available():
    print('usage ERROR: -errtol but exact solution and error not available')
    sys.exit(4)

# set up nested iteration
if args.ni:
    nirange = range(levels)   # hierarchy[j] for j=0,...,levels-1
else:
    nirange = [levels-1,]     # just run on finest level

# fine-mesh initialization
def initial(fmesh, phi, ell):
    if args.exactinitial:
        return obsprob.exact(fmesh.xx())
    else:
        # FIXME create a obsprob method for generating a default initial
        if args.problem == 'poisson':
            # default; sometimes better than phifine when phifine
            #   is negative in places (e.g. -problem parabola)
            return np.maximum(phi, fmesh.zeros())
        elif args.problem == 'sia':
            # FIXME nonzero-thickness initial iterate scheme using
            #   ell (i.e. mass balance)
            return fmesh.zeros()

# nested iteration outer loop
wusum = 0.0
infeascount = 0
actualits = 0
for ni in nirange:
    # evaluate data varphi(x), ell[v] = <f,v> on (current) fine level
    mesh = hierarchy[ni]
    phifine = obsprob.phi(mesh.xx())                   # obstacle
    ellfine = mesh.ellf(obsprob.source(mesh.xx()))     # source functional

    # feasible initial iterate
    if args.ni and ni > 0:
        # prolong and truncate solution from previous coarser level
        uu = np.maximum(phifine, hierarchy[ni].cP(uu))
    else:
        uu = initial(mesh, phifine, ellfine)

    # create monitor using exact solution if available
    uex = None
    if obsprob.exact_available():
        uex = obsprob.exact(mesh.xx())
    mon = ObstacleMonitor(obsprob, mesh,
                          printresiduals=args.monitor, printerrors=args.monitorerr)

    # how many cycles
    if args.ni and ni < levels-1:
        iters = args.nicycles  # use this value if doing nested iteration and
                               #   not yet on finest level
        if args.nicascadic:
            # very simple model for number of cycles before finest
            # compare Blum et al 2004
            iters *= int(np.ceil(1.5**(levels-1-ni)))
    else:
        iters = args.cyclemax

    # multigrid cycles inner loop
    for s in range(iters):
        irnorm, errnorm = mon.irerr(uu, ellfine, phifine, uex=uex, indent=levels-1-ni)
        if errnorm is not None and args.errtol is not None and ni == levels-1:
            # special case: check numerical error on finest level
            if errnorm < args.errtol:
                break
        else:
            # generally stop based on irtol condition
            if s == 0:
                if irnorm == 0.0:
                    break
                irnorm0 = irnorm
            else:
                if irnorm <= args.irtol * irnorm0:
                    break
                if irnorm > 100.0 * irnorm0:
                    print('DIVERGED:  irnorm > 100 irnorm0')
                    break
        if args.sweepsonly:
            # smoother sweeps on finest level
            infeascount += obsprob.smoothersweep(mesh, uu, ellfine, phifine)
            if args.symmetric:
                infeascount += obsprob.smoothersweep(mesh, uu, ellfine, phifine, forward=False)
        else:
            # Tai (2003) constraint decomposition method cycles; default=V(1,0);
            #   Alg. 4.7 in G&K (2009); see mcdl-solver and mcdl-slash in paper
            mesh.chi = phifine - uu                # defect obstacle
            ell = - obsprob.residual(mesh, uu, ellfine)      # base residual
            y, infeas = mcdlcycle(obsprob, ni, hierarchy, ell,
                                  down=args.down, up=args.up, coarse=args.coarse,
                                  levels=levels, view=args.mgview,
                                  symmetric=args.symmetric)
            uu += y
            infeascount += infeas
        actualits = s+1
    else: # if break not called (for loop else)
        mon.irerr(uu, ellfine, phifine, uex=uex, indent=levels-1-ni)

    # accumulate work units from this cycle
    for j in range(ni+1):
        wusum += hierarchy[j].WU / 2**(levels - 1 - j)
        hierarchy[j].WU = 0    # avoids double-counting in nested iteration

# report on computation including numerical error, WU, infeasibles
method = 'using '
if args.ni:
    method = 'nested iter. & '
symstr = 'sym. ' if args.symmetric else ''
if args.sweepsonly:
    method += '%d applications of %ssmoother' % (actualits, symstr)
else:
    method += '%d %sV(%d,%d) cycles' % (actualits, symstr, args.down, args.up)
if obsprob.exact_available():
    uex = obsprob.exact(hierarchy[-1].xx())
    error = ':  |u-uexact|_2 = %.4e' % mesh.l2norm(uu-uex)
else:
    uex = None
    error = ''
countstr = '' if infeascount == 0 else ' (%d infeasibles)' % infeascount
print('fine level %d (m=%d): %s -> %.3f WU%s%s' \
      % (args.jfine, mesh.m, method, wusum, error, countstr))

# graphical output if desired
if args.show or args.o or args.diagnostics:
    vis = VisObstacle(args, obsprob, hierarchy,
                      u=uu, phi=phifine, ell=ellfine, uex=uex)
    vis.generate()
