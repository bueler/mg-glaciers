#!/usr/bin/env python3
'''Solve a 1D obstacle problem by a multilevel constraint decomposition method.'''

# TODO:
#   * correct hier. decomp. figure when up>0

import sys
import argparse
import numpy as np

from meshlevel import MeshLevel1D
from pgs import residual, pgssweep
from subsetdecomp import mcdlcycle
from monitor import ObstacleMonitor
from visualize import VisObstacle

parser = argparse.ArgumentParser(description='''
Solve a 1D obstacle problem:  Find u in
    K = {v in H_0^1[0,1] | v >= phi}
such that the variational inequality holds,
    a(u,v-u) - <f,v-u> >= 0   for all v in K,
where phi is in H_0^1[0,1], f is in L^2[0,1], and
             /1
    a(u,v) = |  u'(x) v'(x) dx.
             /0
Note that the interior condition (PDE) is the Poisson equation  - u'' = f.

Solution is by Alg. 4.7 in Gräser & Kornhuber (2009), namely the multilevel
constraint decomposition slash-cycle method by Tai (2003) in which a monotone
restriction operator decomposes the defect obstacle.  The smoother and the
coarse-mesh solver are projected Gauss-Seidel (PGS).

Option -pgsonly reverts to single-level PGS.

Choose the problem with "-problem icelike" (the default) or "-problem
parabola".  The obstacle can be randomly perturbed with -random.  Choose
the fine-mesh resolution with -kmesh.  Monitor the solution process with
-monitor, -monitorerr, and -mgview.  Get graphical output with -show,
-diagnostics, and -o.

Get other usage help with -h or --help.

References:
  * Gräser, C., & Kornhuber, R. (2009). Multigrid methods for
    obstacle problems. J. Comput. Math. 27 (1), 1--44.
  * Tai, X.-C. (2003). Rate of convergence for some constraint
    decomposition methods for nonlinear variational inequalities.
    Numer. Math. 93 (4), 755--786.
''', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-coarse', type=int, default=1, metavar='N',
                    help='PGS sweeps on coarsest grid (default=1)')
parser.add_argument('-cyclemax', type=int, default=100, metavar='M',
                    help='maximum number of V-cycles (default=100)')
parser.add_argument('-diagnostics', action='store_true', default=False,
                    help='additional diagnostics figures (use with -show or -o)')
parser.add_argument('-down', type=int, default=1, metavar='N',
                    help='PGS sweeps before coarse-mesh correction (default=1)')
parser.add_argument('-errtol', type=float, default=None, metavar='X',
                    help='numerical error should be below this value (default=None)')
parser.add_argument('-fscale', type=float, default=1.0, metavar='X',
                    help='in Poisson equation -u"=f this multiplies f (default X=1)')
parser.add_argument('-irtol', type=float, default=1.0e-3, metavar='X',
                    help='norm of inactive residual is reduced by this factor (default X=10^-3)')
parser.add_argument('-jfine', type=int, default=3, metavar='J',
                    help='fine mesh is jth level (default jfine=3)')
parser.add_argument('-jcoarse', type=int, default=0, metavar='j',
                    help='coarse mesh is jth level (default jcoarse=0 gives 1 node)')
parser.add_argument('-mgview', action='store_true', default=False,
                    help='view multigrid cycles by indented print statements')
parser.add_argument('-monitor', action='store_true', default=False,
                    help='print the inactive-set residual norm after each cycle')
parser.add_argument('-monitorerr', action='store_true', default=False,
                    help='print the error (if available) after each cycle')
parser.add_argument('-ni', action='store_true', default=False,
                    help='use nested iteration for initial iterates (i.e. F-cycle)')
parser.add_argument('-niiters', type=int, default=1, metavar='s',
                    help='nested iteration: iterations on levels before finest (default s=1)')
parser.add_argument('-o', metavar='FILE', type=str, default='',
                    help='save plot at end in image file, e.g. PDF or PNG')
parser.add_argument('-parabolay', type=float, default=-1.0, metavar='X',
                    help='vertical location of obstacle in -problem parabola (default X=-1.0)')
parser.add_argument('-pgsonly', action='store_true', default=False,
                    help='do projected Gauss-Seidel (instead of multigrid)')
parser.add_argument('-plain', action='store_true', default=False,
                    help='when used with -show or -o, only show exact solution and obstacle')
parser.add_argument('-printwarnings', action='store_true', default=False,
                    help='print pointwise feasibility warnings')
parser.add_argument('-problem', choices=['icelike', 'parabola'],
                    metavar='X', default='icelike',
                    help='determines obstacle and source function (default: %(default)s)')
parser.add_argument('-random', action='store_true', default=False,
                    help='make a smooth random perturbation the obstacle')
parser.add_argument('-randomscale', type=float, default=1.0, metavar='X',
                    help='scaling of modes in -random perturbation (default X=1.0)')
parser.add_argument('-randomseed', type=int, default=1, metavar='X',
                    help='seed the generator in -random perturbation (default X=1)')
parser.add_argument('-randommodes', type=int, default=30, metavar='N',
                    help='number of sinusoid modes in -random perturbation (default N=3)')
parser.add_argument('-show', action='store_true', default=False,
                    help='show plot at end')
parser.add_argument('-symmetric', action='store_true', default=False,
                    help='use symmetric projected Gauss-Seidel sweeps (forward then backward)')
parser.add_argument('-up', type=int, default=0, metavar='N',
                    help='PGS sweeps after coarse-mesh correction (default=0; up>0 is V-cycle)')
args, unknown = parser.parse_known_args()

exactavailable = (not args.random) and (args.fscale == 1.0) \
                 and (args.parabolay == -1.0 or args.parabolay <= -2.25)

# provide usage help
if unknown:
    print('usage ERROR: unknown arguments ... try -h or --help for usage')
    sys.exit(1)
if args.show and args.o:
    print('usage ERROR: use either -show or -o FILE but not both')
    sys.exit(2)
if args.monitorerr and not exactavailable:
    print('usage ERROR: -monitorerr but exact solution and error not available')
    sys.exit(3)
if args.errtol is not None and not exactavailable:
    print('usage ERROR: -errtol but exact solution and error not available')
    sys.exit(4)

# fix the random seed for repeatability
np.random.seed(args.randomseed)

def phi(x):
    '''The obstacle:  u >= phi.'''
    if args.problem == 'icelike':
        ph = x * (1.0 - x)
    elif args.problem == 'parabola':
        # maximum is at  2.0 + args.parabolay
        ph = 8.0 * x * (1.0 - x) + args.parabolay
    else:
        raise ValueError
    if args.random:
        perturb = np.zeros(len(x))
        for j in range(args.randommodes):
            perturb += np.random.randn(1) * np.sin((j+1)*np.pi*x)
        perturb *= args.randomscale * 0.03 * np.exp(-10*(x-0.5)**2)
        ph += perturb
    ph[[0, -1]] = [0.0, 0.0]  # always force zero boundary conditions
    return ph

def fsource(x):
    '''The source term in the interior condition -u'' = f.'''
    if args.problem == 'icelike':
        f = 8.0 * np.ones(np.shape(x))
        f[x < 0.2] = -16.0
        f[x > 0.8] = -16.0
    else:
        f = -2.0 * np.ones(np.shape(x))
    return args.fscale * f

def uexact(x):
    '''Assumes x is a numpy array.'''
    assert exactavailable, 'exact solution not available'
    assert args.fscale == 1.0
    if args.problem == 'icelike':
        u = phi(x)
        a, c0, c1, d0, d1 = 0.1, -0.8, 0.09, 4.0, -0.39  # exact values
        mid = (x > 0.2) * (x < 0.8) # logical and
        left = (x > a) * (x < 0.2)
        right = (x > 0.8) * (x < 1.0-a)
        u[mid] = -4.0*x[mid]**2 + d0*x[mid] + d1
        u[left] = 8.0*x[left]**2 + c0*x[left] + c1
        u[right] = 8.0*(1.0-x[right])**2 + c0*(1.0-x[right]) + c1
    else:  # problem == 'parabola'
        if args.parabolay == -1.0:
            a = 1.0/3.0
            def upoisson(x):
                return x * (x - 18.0 * a + 8.0)
            u = phi(x)
            u[x < a] = upoisson(x[x < a])
            u[x > 1.0-a] = upoisson(1.0 - x[x > 1.0-a])
        elif args.parabolay <= -2.25:
            u = x * (x - 1.0)   # solution without obstruction
        else:
            raise NotImplementedError
    return u

# hierarchy is a list of MeshLevel1D with indices [0,..,levels-1]
assert args.jcoarse >= 0
assert args.jfine >= args.jcoarse
levels = args.jfine - args.jcoarse + 1
hierarchy = [None] * (levels)             # list [None,...,None]
for j in range(levels):
    hierarchy[j] = MeshLevel1D(j=j+args.jcoarse)

# nested iteration outer loop
wusum = 0.0
infeascount = 0
if args.ni:
    nirange = range(levels)
else:
    nirange = [levels-1,]     # not nested iteration; just run on finest level
for ni in nirange:
    # evaluate data varphi(x), ell[v] = <f,v> on (current) fine level
    mesh = hierarchy[ni]
    phifine = phi(mesh.xx())                   # obstacle
    ellfine = mesh.ellf(fsource(mesh.xx()))    # source functional

    if args.ni and ni > 0:
        # in nested iteration, prolong solution from previous coarser level,
        #   and truncate for feasible initial iterate
        uu = np.maximum(phifine, hierarchy[ni].cP(uu))
    else:
        # default initial iterate; sometimes better than phifine when phifine
        #   is negative in places (e.g. -problem parabola)
        uu = np.maximum(phifine, mesh.zeros())

    # create monitor; use exact solution if available
    uex = None
    if exactavailable:
        uex = uexact(mesh.xx())
    mon = ObstacleMonitor(mesh, ellfine, phifine, uex=uex,
                          printresiduals=args.monitor, printerrors=args.monitorerr)

    # multigrid slash-cycles or V-cycles inner loop
    if args.ni and ni < levels-1:
        iters = args.niiters
    else:
        iters = args.cyclemax
    for s in range(iters):
        irnorm, errnorm = mon.irerr(uu, indent=levels-1-ni)
        if errnorm is not None and args.errtol is not None and ni == levels-1:
            # special case: check numerical error on finest level
            if errnorm < args.errtol:
                break
        else:
            # generally stop based on irtol condition
            if irnorm < 1.0e-50:
                break
            if s == 0:
                irnorm0 = irnorm
            else:
                if irnorm < args.irtol * irnorm0:
                    break
        if args.pgsonly:
            # revert to sweeps of projected Gauss-Seidel on fine grid
            infeascount += pgssweep(mesh, uu, ellfine, phifine,
                                    printwarnings=args.printwarnings)
            if args.symmetric:
                infeascount += pgssweep(mesh, uu, ellfine, phifine,
                                        forward=False, printwarnings=args.printwarnings)
        else:
            # Tai (2003) constraint decomposition method; usually V(1,0)-cycles;
            #   Alg. 4.7 in G&K (2009); next lines are "mcdl-solver()" in paper
            mesh.chi = phifine - uu                # defect obstacle
            ell = - residual(mesh,uu,ellfine)      # base residual
            y, infeas = mcdlcycle(ni, hierarchy, ell,
                                  down=args.down, up=args.up, coarse=args.coarse,
                                  levels=levels, view=args.mgview,
                                  symmetric=args.symmetric,
                                  printwarnings=args.printwarnings)
            uu += y
            infeascount += infeas

    # finalize iterations and monitor (see stopping criterion above)
    its = s
    if s == iters - 1:
        mon.irerr(uu, indent=levels-1-ni)
        its = s+1

    # accumulate work units from this slash
    for j in range(ni+1):
        wusum += hierarchy[j].WU / 2**(levels - 1 - j)
        hierarchy[j].WU = 0    # avoids double-counting in nested iteration

# report on computation; includes numerical error, WU, infeasibles
method = 'using '
if args.ni:
    method = 'nested iter. & '
symstr = 'sym. ' if args.symmetric else ''
if args.pgsonly:
    method += '%d applications of %sPGS' % (its, symstr)
else:
    method += '%d %sV(%d,%d) cycles' % (its, symstr, args.down, args.up)
if exactavailable:
    uex = uexact(hierarchy[-1].xx())
    error = ':  |u-uexact|_2 = %.4e' % mesh.l2norm(uu-uex)
else:
    uex = None
    error = ''
countstr = '' if infeascount == 0 else ' (%d infeasibles)' % infeascount
print('fine level %d (m=%d): %s -> %.3f WU%s%s' \
      % (args.jfine, mesh.m, method, wusum, error, countstr))

# graphical output if desired
if args.show or args.o or args.diagnostics:
    vis = VisObstacle(mesh, phifine)
    if args.show or args.o:
        if args.plain:
            if exactavailable:
                vis.plain(uex, filename=args.o)
            else:
                raise ValueError('graphic (-plain) not available if no exact solution')
        else:
            vis.final(uu, filename=args.o, uex=uex)
    if args.diagnostics:
        if len(args.o) > 0:
            rname = 'resid_' + args.o
        else:
            rname = ''
        vis.residuals(uu, ellfine, filename=rname)
        if not args.pgsonly:
            if len(args.o) > 0:
                dname = 'decomp_' + args.o
                iname = 'icedec_' + args.o
            else:
                dname, iname = '', ''
            vis.decomposition(hierarchy, up=args.up, filename=dname)
            vis.icedecomposition(hierarchy, phifine, up=args.up, filename=iname)
