#!/usr/bin/env python3

import numpy as np
import sys, argparse
import matplotlib.pyplot as plt

from meshlevel import MeshLevel
from monotonetai import pgssweep,vcycle

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
parser.add_argument('-downsweeps', type=int, default=1, metavar='N',
                    help='number of sweeps of projected Gauss-Seidel (default=1)')
parser.add_argument('-fvalue', type=float, default=-2.0, metavar='X',
                    help='in Poisson equation -u"=f this sets the constant f-value (default X=-2)')
parser.add_argument('-jfine', type=int, default=2, metavar='J',
                    help='fine mesh is jth level (default jfine=2)')
parser.add_argument('-low', action='store_true', default=False,
                    help='use obstacle sufficiently low to have no contact')
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
parser.add_argument('-random', action='store_true', default=False,
                    help='use obstacle which is random perturbation of usual')
parser.add_argument('-randommag', type=float, default=0.4, metavar='X',
                    help='magnitude of -random perturbation (default X=0.4)')
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
    ph = 8.0 * x * (1.0 - x) - 1.0
    if args.low:
        ph -= 9.0
    if args.random:
        ph += args.randommag * np.random.rand(len(x))
    return ph

def fsource(x):
    '''The source term in -u'' = f.'''
    return args.fvalue * np.ones(np.shape(x))

def fzero(x):
    return np.zeros(np.shape(x))

def uexact(x):
    '''A by-hand calculation from -u''=-2, and given the above obstacle, shows
    that u(a)=phi(a), u'(a)=phi'(a) for a<1/2 implies a=1/3.'''
    if args.low:
        u = x * (x - 1.0)   # solution without obstruction
    else:
        a = 1.0/3.0
        def upoisson(x):
            return x * (x - 18.0 * a + 8.0)
        u = phi(x)
        u[x < a] = upoisson(x[x < a])
        u[x > 1.0-a] = upoisson(1.0 - x[x > 1.0-a])
    return u

def finalplot(xx,uinitial,ufinal,phifine):
    plt.figure(figsize=(15.0,8.0))
    plt.plot(xx,uinitial,'k--',label='initial iterate',linewidth=2.0)
    plt.plot(xx,ufinal,'k',label='final iterate',linewidth=4.0)
    plt.plot(xx,phifine,'r',label='obstacle',linewidth=2.0)
    if not args.random:
        plt.plot(xx,uexact(xx),'g',label='exact',linewidth=2.0)
    plt.axis([0.0,1.0,-0.3,1.1*max(phifine)])
    plt.legend(fontsize=20.0)
    plt.xlabel('x')

# mesh hierarchy = [coarse,...,fine]
levels = args.jfine - args.jcoarse + 1
hierarchy = [None] * (levels)  # list [None,...,None]
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

# monotone multigrid V-cycles (unless user just wants pGS)
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
        if args.monitor:
            print('  %d:  |u-uexact|_2 = %.4e' % (s,l2err(uu)))
        uu = vcycle(uu,phifine,fsource,hierarchy,
                    levels=levels,view=args.mgview,symmetric=args.symmetric,
                    downsweeps=args.downsweeps,
                    coarsesweeps=args.coarsesweeps,
                    upsweeps=args.upsweeps)

# report on computation including numerical error
if args.pgs:
   method = 'with %d sweeps of pGS' % args.downsweeps
else:
   method = 'using %d V(%d,%d,%d) cycles' \
            % (args.cycles,args.downsweeps,args.coarsesweeps,args.upsweeps)
if args.random:
   error = ''
else:
   error = ':  |u-uexact|_2 = %.4e' % l2err(uu)
print('level %d (m = %d) %s%s' % (args.jfine,mesh.m,method,error))

# graphical output if desired
if args.show or args.o:
    finalplot(mesh.xx(),uinitial,uu,phifine)

    # FIXME put option control on this figure which shows defect hierarchy
    if False:
        plt.figure(figsize=(15.0,8.0))
        for k in range(args.j+1):
            plt.plot(hierarchy[k].xx(),chi[k],'k.-')  # FIXME chi[] is now inside vcycle()
        plt.xlabel('x')

    if args.o:
        plt.savefig(args.o,bbox_inches='tight')
    else:
        plt.show()

