#!/usr/bin/env python3

import numpy as np
import sys, argparse
import matplotlib.pyplot as plt

from meshlevel import MeshLevel
from subsetdecomp import pgssweep,vcycle

parser = argparse.ArgumentParser(description='''
Solve a 1D obstacle problem:
                     /1
    min_{u >= phi}   |  1/2 (u')^2 - f u
                     /0
where phi(x) = 8x(1-x)-1, f(x) = -2, and u is in H_0^1[0,1].
Interior condition is -u''=-2.

Solution by Alg. 4.7 in Gräser & Kornhuber (2009), monotone multigrid
V-cycles using the Tai (2003) method.  The smoother and the coarse-mesh
solver is projected Gauss-Seidel (pGS).  Monotone restrictions decompose
the defect obstacle.  Option -pgs reverts to single-level pGS (using
-downsweeps).

References:
* Gräser, C., & Kornhuber, R. (2009). Multigrid methods for
obstacle problems. J. Comput. Math. 27 (1), 1--44.
* Tai, X.-C. (2003). Rate of convergence for some constraint
decomposition methods for nonlinear variational inequalities.
Numer. Math. 93 (4), 755--786.
''',add_help=False,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-coarsesweeps', type=int, default=1, metavar='N',
                    help='number of sweeps of projected Gauss-Seidel (default=1)')
parser.add_argument('-cycles', type=int, default=2, metavar='M',
                    help='number of V-cycles (default=2)')
parser.add_argument('-diagnostics', action='store_true', default=False,
                    help='add a diagnostics figure to -show or -o output')
parser.add_argument('-downsweeps', type=int, default=1, metavar='N',
                    help='number of sweeps of projected Gauss-Seidel (default=1)')
parser.add_argument('-fscale', type=float, default=1.0, metavar='X',
                    help='in Poisson equation -u"=f this multiplies f (default X=1)')
parser.add_argument('-jfine', type=int, default=2, metavar='J',
                    help='fine mesh is jth level (default jfine=2)')
parser.add_argument('-jcoarse', type=int, default=0, metavar='J',
                    help='coarse mesh is jth level (default jcoarse=0 gives 1 node)')
parser.add_argument('-mgview', action='store_true', default=False,
                    help='view multigrid cycles by indented print statements')
parser.add_argument('-monitor', action='store_true', default=False,
                    help='monitor the error at the end of each multigrid cycle')
parser.add_argument('-obs1help', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-o', metavar='FILE', type=str, default='',
                    help='save plot at end in image file, e.g. PDF or PNG')
parser.add_argument('-pgs', action='store_true', default=False,
                    help='do projected Gauss-Seidel (instead of multigrid)')
parser.add_argument('-problem', choices=['parabola', 'low', 'icelike'],
                    metavar='X', default='parabola',
                    help='determines obstacle and source function (default: %(default)s)')
parser.add_argument('-random', action='store_true', default=False,
                    help='randomly perturb the obstacle')
parser.add_argument('-randommag', type=float, default=0.2, metavar='X',
                    help='magnitude of -random perturbation (default X=0.2)')
parser.add_argument('-show', action='store_true', default=False,
                    help='show plot at end')
parser.add_argument('-symmetric', action='store_true', default=False,
                    help='use symmetric projected Gauss-Seidel sweeps (forward then backward)')
parser.add_argument('-upsweeps', type=int, default=0, metavar='N',
                    help='number of sweeps of projected Gauss-Seidel (default=1)')
args, unknown = parser.parse_known_args()
if args.obs1help:
    parser.print_help()
    sys.exit(0)

# fix the random seed for repeatability
np.random.seed(1)

def phi(x):
    '''The obstacle:  u >= phi.'''
    if args.problem == 'parabola':
        ph = 8.0 * x * (1.0 - x) - 1.0
    elif args.problem == 'low':
        ph = 8.0 * x * (1.0 - x) - 3.0
    elif args.problem == 'icelike':
        ph = 4.0 * x * (1.0 - x)
    else:
        raise ValueError
    if args.random:
        perturb = args.randommag * np.random.randn(len(x))
        perturb[0] = 0.0
        perturb[-1] = 0.0
        ph += perturb
    return ph

def fsource(x):
    '''The source term in -u'' = f.  Assumes x is scalar.'''
    if args.problem == 'icelike':
        if x < 0.3 or x > 0.7:
            f = -5.0
        else:
            f = 20.0
    else:
        f = -2.0
    return args.fscale * f

def fzero(x):
    return np.zeros(np.shape(x))

exactavailable = (args.problem == 'parabola' or args.problem == 'low') \
                 and (not args.random) and (args.fscale == 1.0)

# FIXME exact solution for icelike
def uexact(x):
    assert exactavailable, 'exact solution not actually available'
    if args.problem == 'low':
        u = x * (x - 1.0)   # solution without obstruction
    else:
        a = 1.0/3.0
        def upoisson(x):
            return x * (x - 18.0 * a + 8.0)
        u = phi(x)
        u[x < a] = upoisson(x[x < a])
        u[x > 1.0-a] = upoisson(1.0 - x[x > 1.0-a])
    return u

# mesh hierarchy = [coarse,...,fine]
levels = args.jfine - args.jcoarse + 1
hierarchy = [None] * (levels)  # list [None,...,None]; indices 0,...,levels-1
for k in range(args.jcoarse,args.jfine+1):
   hierarchy[k-args.jcoarse] = MeshLevel(k=k)
mesh = hierarchy[-1]  # fine mesh

# discrete obstacle on fine level
phifine = phi(mesh.xx())

# feasible initial iterate
uinitial = np.maximum(phifine,np.zeros(np.shape(mesh.xx())))

# ability to evaluate error
def l2err(u):
    udiff = u - uexact(mesh.xx())
    return mesh.l2norm(udiff)

# multigrid V-cycles (unless user just wants pGS)
uu = uinitial.copy()
if args.pgs:
    # sweeps of projected Gauss-Seidel on fine grid
    r = mesh.residual(mesh.zeros(),fsource)
    for s in range(args.downsweeps):
        pgssweep(mesh.m,mesh.h,uu,r,phifine)
        if args.symmetric:
            pgssweep(mesh.m,mesh.h,uu,r,phifine,backward=True)
else:
    for s in range(args.cycles):
        if args.monitor and exactavailable:
            print('  %d:  |u-uexact|_2 = %.4e' % (s,l2err(uu)))
        uu, chi = vcycle(uu,phifine,fsource,hierarchy,
                         levels=levels,view=args.mgview,
                         symmetric=args.symmetric,
                         downsweeps=args.downsweeps,
                         coarsesweeps=args.coarsesweeps,
                         upsweeps=args.upsweeps)

# report on computation including numerical error
if args.pgs:
   method = 'with %d sweeps of pGS' % args.downsweeps
else:
   method = 'using %d V(%d,%d,%d) cycles' \
            % (args.cycles,args.downsweeps,args.coarsesweeps,args.upsweeps)
if exactavailable:
   error = ':  |u-uexact|_2 = %.4e' % l2err(uu)
else:
   error = ''
print('level %d (m = %d) %s%s' % (args.jfine,mesh.m,method,error))

# graphical output if desired
import matplotlib
font = {'size' : 20}
matplotlib.rc('font', **font)
lines = {'linewidth': 2}
matplotlib.rc('lines', **lines)

def finalplot(xx,uinitial,ufinal,phifine):
    plt.figure(figsize=(15.0,8.0))
    plt.plot(xx,uinitial,'k--',label='initial iterate')
    plt.plot(xx,ufinal,'k',label='final iterate',linewidth=4.0)
    plt.plot(xx,phifine,'r',label='obstacle')
    if exactavailable:
        plt.plot(xx,uexact(xx),'g',label='exact')
    plt.axis([0.0,1.0,-0.3 + min(ufinal),1.1*max(ufinal)])
    plt.legend()
    plt.xlabel('x')

if args.show or args.o:
    finalplot(mesh.xx(),uinitial,uu,phifine)
    if args.o:
        plt.savefig(args.o,bbox_inches='tight')
    else:
        plt.show()

if args.diagnostics:
    plt.figure(figsize=(15.0,15.0))
    plt.subplot(4,1,1)
    r = mesh.residual(uu,fsource)
    osr = mesh.inactiveresidual(uu,fsource,phifine)
    plt.plot(mesh.xx(),r,'k',label='residual')
    plt.plot(mesh.xx(),osr,'r',label='inactive residual')
    plt.legend()
    plt.gca().set_xticks([],[])
    plt.subplot(4,1,2)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(mesh.xx(),osr,'r',label='inactive residual')
    plt.legend()
    plt.gca().set_xticks([],[])
    plt.subplot(4,1,(3,4))
    for k in range(levels-1):
        plt.plot(hierarchy[k].xx(),chi[k],'k.--',ms=8.0,
                 label='level %d' % k)
    plt.plot(hierarchy[levels-1].xx(),chi[levels-1],'k.-',ms=12.0,
             label='fine mesh',linewidth=3.0)
    plt.legend()
    plt.title('decomposition of final defect obstacle')
    plt.xlabel('x')
    if args.o:
        plt.savefig('diags_' + args.o,bbox_inches='tight')
    else:
        plt.show()

