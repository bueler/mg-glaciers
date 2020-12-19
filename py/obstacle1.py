#!/usr/bin/env python3

import numpy as np
import sys, argparse
import matplotlib.pyplot as plt

from meshlevel import MeshLevel

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
the defect obstacle.  Option -pgs reverts to single-level pGS.

References:
* Gräser, C., & Kornhuber, R. (2009). Multigrid methods for
obstacle problems. J. Comput. Math. 27 (1), 1--44.
* Tai, X.-C. (2003). Rate of convergence for some constraint
decomposition methods for nonlinear variational inequalities.
Numer. Math. 93 (4), 755--786.
''',add_help=False,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-cycles', type=int, default=2, metavar='M',
                    help='number of V-cycles (default=2)')
parser.add_argument('-j', type=int, default=2, metavar='J',
                    help='fine grid level (default j=2 gives 8 subintervals)')
parser.add_argument('-lowobstacle', action='store_true', default=False,
                    help='use obstacle sufficiently low to have no contact')
parser.add_argument('-obstacle1help', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-o', metavar='FILE', type=str, default='',
                    help='save plot at end in image file, e.g. PDF or PNG')
parser.add_argument('-pgs', action='store_true', default=False,
                    help='do projected Gauss-Seidel (instead of multigrid)')
parser.add_argument('-show', action='store_true', default=False,
                    help='show plot at end')
parser.add_argument('-sweeps', type=int, default=1, metavar='N',
                    help='number of sweeps of projected Gauss-Seidel (default=1)')
parser.add_argument('-mgview', action='store_true', default=False,
                    help='view multigrid cycles by indented print statements')
args, unknown = parser.parse_known_args()
if args.obstacle1help:
    parser.print_help()
    sys.exit(0)

# FIXME add option for random-ish fine-mesh obstacle
def phi(low,x):
    '''The obstacle:  u >= phi.'''
    if low:
        return 8.0 * x * (1.0 - x) - 10.0
    else:
        return 8.0 * x * (1.0 - x) - 1.0

def fsource(x):
    '''The source term in -u'' = f.'''
    return - 2.0 * np.ones(np.shape(x))

def fzero(x):
    '''The zero function.'''
    return np.zeros(np.shape(x))

def uexact(low,x):
    '''A by-hand calculation from -u''=-2, and given the above obstacle, shows
    that u(a)=phi(a), u'(a)=phi'(a) for a<1/2 implies a=1/3.'''
    if low:
        u = x * (x - 1.0)   # solution without obstruction
    else:
        a = 1.0/3.0
        def upoisson(x):
            return x * (x - 18.0 * a + 8.0)
        u = phi(False,x)
        u[x < a] = upoisson(x[x < a])
        u[x > 1.0-a] = upoisson(1.0 - x[x > 1.0-a])
    return u

def finalplot(xx,uinitial,ufinal):
    plt.figure(figsize=(15.0,8.0))
    plt.plot(xx,uinitial,'k--',label='initial iterate',linewidth=2.0)
    plt.plot(xx,ufinal,'k',label='final iterate',linewidth=4.0)
    plt.plot(xx,phi(args.lowobstacle,xx),'r',label='obstacle',linewidth=2.0)
    plt.plot(xx,uexact(args.lowobstacle,xx),'g',label='exact',linewidth=2.0)
    plt.axis([0.0,1.0,-0.3,1.1])
    plt.legend(fontsize=20.0)
    plt.xlabel('x')

# mesh hierarchy = [coarse,...,fine]
# FIXME allow coarse grid to be of any resolution
hierarchy = [None] * (args.j+1)  # list [None,...,None] for indices 0,1,...,j
for k in range(args.j+1):
   hierarchy[k] = MeshLevel(k=k,f=fsource)
mesh = hierarchy[-1]  # fine mesh

# discrete obstacle on fine level
phifine = phi(args.lowobstacle,mesh.xx)

# feasible initial iterate
uinitial = np.maximum(phifine,np.zeros(np.shape(mesh.xx)))

def indentprint(n,s):
    '''Indent n levels and print string s.'''
    #for i in range(n):
    #    print('  ',end='')
    print(s)

def vcycle(hierarchy,phifine,uinitial):
    # Algorithm 4.7 in G&K
    # FIXME indent printing and make it optional
    # SETUP
    chi = [None] * (args.j+1)         # empty list of length j+1
    chi[args.j] = phifine - uinitial  # fine mesh defect obstacle
    uu = uinitial.copy()
    r = mesh.residual(uu)             # fine mesh residual
    # DOWN-SMOOTH
    for k in range(args.j,0,-1):
        # monotone restriction gives defect obstacle (G&K formulas after (4.22))
        chi[k-1] = hierarchy[k].MRO(chi[k])
        # defect obstacle change on mesh k
        psi = chi[k] - hierarchy[k].prolong(chi[k-1])
        # do projected GS sweeps (FIXME try symmetric)
        hk = hierarchy[k].h
        v = hierarchy[k].zeros()
        if args.mgview:
            indentprint(args.j-k,'mesh %d: %d sweeps over %d interior points' \
                                 % (k,args.sweeps,hierarchy[k].m-1))
        for s in range(args.sweeps):
            hierarchy[k].pgssweep(v,r=r,phi=psi)
        hierarchy[k].vstate = v.copy()
        # update the residual and canonically-restrict it
        r += hierarchy[k].residual(v,f=fzero)
        r = hierarchy[k].CR(r)
    # COARSE SOLVE using fixed number of projected GS sweeps
    psi = chi[0]
    h0 = hierarchy[0].h
    v = hierarchy[0].zeros()
    if args.mgview:
        indentprint(args.j,'coarse: %d sweeps over %d interior points' \
                           % (args.sweeps,hierarchy[0].m-1))
    for s in range(args.sweeps):
        hierarchy[0].pgssweep(v,r=r,phi=psi)
    hierarchy[0].vstate = v.copy()
    # UP (WITHOUT SMOOTHING)
    # FIXME allow up-smoothing
    for k in range(1,args.j+1):
        if args.mgview:
            indentprint(args.j-k,'mesh %d: interpolate up to %d interior points' \
                                 % (k,hierarchy[k].m-1))
        hierarchy[k].vstate += hierarchy[k].prolong(hierarchy[k-1].vstate)
    # FINALIZE
    uu += mesh.vstate
    return uu

# monotone multigrid V-cycles (unless user just wants pGS)
# FIXME allow more than one
if args.pgs:
    # sweeps of projected Gauss-Seidel on fine grid
    uu = uinitial.copy()
    r = mesh.residual(mesh.zeros())
    for s in range(args.sweeps):
        mesh.pgssweep(uu,r=r,phi=phifine)
else:
    uu = vcycle(hierarchy,phifine,uinitial)

# evaluate numerical error
udiff = uu - uexact(args.lowobstacle,mesh.xx)
#FIXME remove indent
print('  level %d (m = %d) using %d sweeps:  |u-uexact|_2 = %.4e' \
      % (args.j,mesh.m,args.sweeps,mesh.l2norm(udiff)))

# graphical output if desired
if args.show or args.o:
    finalplot(mesh.xx,uinitial,uu)

    # FIXME put option control on this figure which shows defect hierarchy
    if False:
        plt.figure(figsize=(15.0,8.0))
        for k in range(args.j+1):
            plt.plot(hierarchy[k].xx,chi[k],'k.-')
        plt.xlabel('x')

    if args.o:
        plt.savefig(args.o,bbox_inches='tight')
    else:
        plt.show()

