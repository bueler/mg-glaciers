#!/usr/bin/env python3

# TODO: count WU

import numpy as np
import sys, argparse

from meshlevel import MeshLevel1D
from poisson import ellf
from pgs import inactiveresidual, pgssweep
from subsetdecomp import vcycle
from visualize import obstacleplot, obstaclediagnostics

parser = argparse.ArgumentParser(description='''
Solve a 1D obstacle problem:
                     /1
    min_{u >= phi}   |  1/2 (u')^2 - f u
                     /0
where phi(x) and u(x) are in H_0^1[0,1] and f(x) is in L^2[0,1].
Note that the interior condition (PDE) is  - u'' = f.

Solution is by Alg. 4.7 in Gräser & Kornhuber (2009), namely the subset
decomposition V-cycle method by Tai (2003).  The smoother and the coarse-mesh
solver are projected Gauss-Seidel (pGS).  Monotone restrictions decompose
the defect obstacle.  Option -pgsonly reverts to single-level pGS.

Get usage help with -h or --help.

References:
* Gräser, C., & Kornhuber, R. (2009). Multigrid methods for
obstacle problems. J. Comput. Math. 27 (1), 1--44.
* Tai, X.-C. (2003). Rate of convergence for some constraint
decomposition methods for nonlinear variational inequalities.
Numer. Math. 93 (4), 755--786.
''',formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-coarse', type=int, default=1, metavar='N',
                    help='pGS sweeps on coarsest grid (default=1)')
parser.add_argument('-cyclemax', type=int, default=100, metavar='M',
                    help='maximum number of V-cycles (default=100)')
parser.add_argument('-diagnostics', action='store_true', default=False,
                    help='add a diagnostics figure to -show or -o output')
parser.add_argument('-down', type=int, default=1, metavar='N',
                    help='pGS sweeps before coarse-mesh correction (default=1)')
parser.add_argument('-fscale', type=float, default=1.0, metavar='X',
                    help='in Poisson equation -u"=f this multiplies f (default X=1)')
parser.add_argument('-irtol', type=float, default=1.0e-3, metavar='X',
                    help='norm of inactive residual is reduced by this factor (default X=10^-3)')
parser.add_argument('-kfine', type=int, default=3, metavar='K',
                    help='fine mesh is kth level (default kfine=3)')
parser.add_argument('-kcoarse', type=int, default=0, metavar='k',
                    help='coarse mesh is kth level (default kcoarse=0 gives 1 node)')
parser.add_argument('-mgview', action='store_true', default=False,
                    help='view multigrid cycles by indented print statements')
parser.add_argument('-monitor', action='store_true', default=False,
                    help='monitor the inactive residual norm at the end of each V-cycle')
parser.add_argument('-monitorerr', action='store_true', default=False,
                    help='monitor the error (if available) at the end of each V-cycle')
parser.add_argument('-o', metavar='FILE', type=str, default='',
                    help='save plot at end in image file, e.g. PDF or PNG')
parser.add_argument('-parabolay', type=float, default=-1.0, metavar='X',
                    help='vertical location of obstacle in -problem parabola (default X=-1.0)')
parser.add_argument('-pgsonly', action='store_true', default=False,
                    help='do projected Gauss-Seidel (instead of multigrid)')
parser.add_argument('-problem', choices=['icelike','parabola'],
                    metavar='X', default='icelike',
                    help='determines obstacle and source function (default: %(default)s)')
parser.add_argument('-random', action='store_true', default=False,
                    help='make a smooth random perturbation the obstacle')
parser.add_argument('-randomscale', type=float, default=1.0, metavar='X',
                    help='scaling of modes in -random perturbation (default X=1.0)')
parser.add_argument('-randommodes', type=int, default=30, metavar='N',
                    help='number of sinusoid modes in -random perturbation (default N=3)')
parser.add_argument('-show', action='store_true', default=False,
                    help='show plot at end')
parser.add_argument('-symmetric', action='store_true', default=False,
                    help='use symmetric projected Gauss-Seidel sweeps (forward then backward)')
parser.add_argument('-up', type=int, default=1, metavar='N',
                    help='pGS sweeps after coarse-mesh correction (default=1)')
args, unknown = parser.parse_known_args()

# FIXME decide which parabolay values allow exact
exactavailable = (not args.random) and (args.fscale == 1.0)

# provide usage help
if unknown:
    print('ERROR: unknown arguments ... try -h or --help for usage')
    sys.exit(1)
if args.show and args.o:
    print('ERROR: use either -show or -o FILE but not both')
    sys.exit(2)
if args.monitorerr and not exactavailable:
    print('ERROR: -monitorerr but exact solution and error not available')
    sys.exit(3)

# fix the random seed for repeatability
np.random.seed(1)

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
    ph[[0,-1]] = [0.0, 0.0]  # always force zero boundary conditions
    return ph

def fsource(x):
    '''The source term in -u'' = f.'''
    if args.problem == 'icelike':
        f = 8.0 * np.ones(np.shape(x))
        f[x<0.2] = -16.0
        f[x>0.8] = -16.0
    else:
        f = -2.0 * np.ones(np.shape(x))
    return args.fscale * f

def uexact(x):
    '''Assumes x is a numpy array.'''
    assert exactavailable, 'exact solution not available'
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

# mesh hierarchy = [coarse,...,fine]
assert (args.kcoarse >= 0)
assert (args.kfine >= args.kcoarse)
levels = args.kfine - args.kcoarse + 1
assert (levels >= 1)
hierarchy = [None] * (levels)  # list [None,...,None]; indices 0,...,levels-1
for k in range(levels):
   hierarchy[k] = MeshLevel1D(k=k+args.kcoarse)
mesh = hierarchy[-1]  # fine mesh

# discrete obstacle on fine level
phifine = phi(mesh.xx())

# feasible initial iterate
uinitial = np.maximum(phifine,mesh.zeros())
uinitial[[0,-1]] = [0.0,0.0]

# exact solution
if exactavailable:
    uex = uexact(mesh.xx())

# multigrid V-cycles (unless user just wants pGS)
uu = uinitial.copy()
ellfine = ellf(mesh,fsource(mesh.xx()))
if args.pgsonly:
    # sweeps of projected Gauss-Seidel on fine grid
    for s in range(args.cyclemax):
        ir = mesh.l2norm(inactiveresidual(mesh,uu,ellfine,phifine))
        if ir < 1.0e-50:
            break
        if s == 0:
            ir0 = ir
        else:
            if ir < args.irtol * ir0 or ir < 1.0e-50:
                break
        pgssweep(mesh,uu,ellfine,phifine)
        if args.symmetric:
            pgssweep(mesh,uu,ellfine,phifine,forward=False)
else:
    for s in range(args.cyclemax):
        ir = mesh.l2norm(inactiveresidual(mesh,uu,ellfine,phifine))
        if ir < 1.0e-50:
            break
        if s == 0:
            ir0 = ir
        else:
            if ir < args.irtol * ir0:
                break
        if args.monitor:
            print('  %d:  |r^i(u)|_2 = %.4e' % (s,ir))
        if args.monitorerr and exactavailable:
            print('  %d:  |u-uexact|_2 = %.4e' % (s,mesh.l2norm(uu-uex)))
        uu, chi = vcycle(hierarchy,uu,ellfine,phifine,
                         levels=levels,view=args.mgview,symmetric=args.symmetric,
                         down=args.down,coarse=args.coarse,up=args.up)
if args.monitor:
    ir = mesh.l2norm(inactiveresidual(mesh,uu,ellfine,phifine))
    print('  %d:  |r^i(u)|_2 = %.4e' % (s+1,ir))

# report on computation including numerical error
if args.pgsonly:
   method = 'with %d applications of pGS' % (s+1)
else:
   method = 'using %d V(%d,%d,%d) cycles' \
            % (s+1,args.down,args.coarse,args.up)
if exactavailable:
   error = ':  |u-uexact|_2 = %.4e' % mesh.l2norm(uu-uex)
else:
   uex = []
   error = ''
print('fine level %d (m = %d) %s%s' % (args.kfine,mesh.m,method,error))

# graphical output if desired
if args.show or args.o:
    obstacleplot(mesh,uinitial,uu,phifine,args.o,uex=uex)
if args.diagnostics:
    obstaclediagnostics(hierarchy,uu,phifine,ellfine,chi,args.up,args.o)
