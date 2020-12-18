#!/usr/bin/env python3

import numpy as np
import sys, argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='''
Solve a 1D obstacle problem:
                     /1
    min_{u >= phi}   |  1/2 (u')^2 - f u
                     /0
where  phi(x) = 8x(1-x)-1,  f(x) = -2,  and  u in H_0^1[0,1].
(Thus where u>phi we have -u''=-2.)

The solution method is Algorithm 4.7 in Gräser & Kornhuber (2009),
that is, by monotone multigrid V cycles using the Tai (2003)
method.  The smoother at each level is projected Gauss-Seidel (pGS),
with monotone restrictions to generate a hierarchical decomposition
of the defect obstacle.  Option -pgs reverts to single-level pGS.

Reference: Gräser, C., & Kornhuber, R. (2009). Multigrid methods for
obstacle problems. J. Comput. Math., 1-44.
''',add_help=False)
parser.add_argument('-pgs', action='store_true', default=False,
                    help='do projected Gauss-Seidel (instead of multigrid)')
parser.add_argument('-j', type=int, default=2, metavar='J',
                    help='fine grid level (default j=2 gives 8 subintervals)')
parser.add_argument('-obstacle1help', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-o', metavar='FILE.pdf', type=str, default='',
                    help='save plot at end in image file')
parser.add_argument('-show', action='store_true', default=False,
                    help='show plot at end')
parser.add_argument('-sweeps', type=int, default=1, metavar='N',
                    help='number of sweeps of projected Gauss-Seidel (default=1)')
args, unknown = parser.parse_known_args()
if args.obstacle1help:
    parser.print_help()
    sys.exit(0)

def phi(x):
    '''The obstacle:  u >= phi.'''
    return 8.0 * x * (1.0 - x) - 1.0

def fsource(x):
    '''The source term in -u'' = f.'''
    return - 2.0 * np.ones(np.shape(x))

def fzero(x):
    '''The zero function.'''
    return np.zeros(np.shape(x))

def uexact(x):
    '''A by-hand calculation from -u''=-2, and given the above obstacle, shows
    that u(a)=phi(a), u'(a)=phi'(a) for a<1/2 implies a=1/3.'''
    a = 1.0/3.0
    def upoisson(x):
        return x * (x - 18.0 * a + 8.0)
    u = phi(x)
    u[x < a] = upoisson(x[x < a])
    u[x > 1.0-a] = upoisson(1.0 - x[x > 1.0-a])
    return u

def finalplot(xx,uinitial,ufinal):
    plt.figure(figsize=(15.0,8.0))
    plt.plot(xx,uinitial,'k--',label='initial iterate',linewidth=2.0)
    plt.plot(xx,ufinal,'k',label='final iterate',linewidth=4.0)
    plt.plot(xx,phi(xx),'r',label='obstacle',linewidth=2.0)
    plt.plot(xx,uexact(xx),'g',label='exact',linewidth=2.0)
    plt.axis([0.0,1.0,-0.3,1.1])
    plt.legend(fontsize=20.0)
    plt.xlabel('x')

class MeshLevel(object):
    '''Encapsulate a mesh level for the interval [0,1].  MeshLevel(k=0)
    is the coarse mesh and MeshLevel(k=j) is the fine mesh.
    MeshLevel(k=k) has m = 2^{k+1} subintervals.  This object knows
    about zero vectors, L_2 norms, residuals, prolongation,
    canonical restriction, and monotone restriction.'''

    def __init__(self, k=None):
        self.k = k
        self.m = 2**(self.k+1)
        self.mcoarser = 2**self.k
        self.h = 1.0 / self.m
        self.xx = np.linspace(0.0,1.0,self.m+1)  # indices 0,1,...,m
        self.vstate = None

    def zeros(self):
        return np.zeros(self.m+1)

    def l2norm(self, u):
        '''L^2[0,1] norm computed with trapezoid rule.'''
        return np.sqrt(self.h * (0.5*u[0]*u[0] + np.sum(u[1:-1]*u[1:-1]) \
                                 + 0.5*u[-1]*u[-1]))

    def residual(self,f,u):
        '''Represent the residual linear functional (i.e. in S_k')
           r(v) = ell(v) - a(u,v)
                = int_0^1 f v - int_0^1 u' v'
        associated to state u by a vector r for the interior points.
        Returned r satisfies r[0]=0 and r[m]=0.  Uses midpoint rule
        for the first integral and the exact value for the second.'''
        assert len(u) == self.m+1, \
               'input vector of length %d (should be %d)' % (len(v),self.m+1)
        r = self.zeros()
        for p in range(1,self.m):
            xpm, xpp = (p-0.5) * self.h, (p+0.5) * self.h
            r[p] = (self.h/2.0) * (f(xpm) + f(xpp)) \
                   - (1.0/self.h) * (2.0*u[p] - u[p-1] - u[p+1])
        return r

    def prolong(self,v):
        '''Prolong a vector on the next-coarser (k-1) mesh (i.e.
        in S_{k-1}) onto the current mesh (in S_k).'''
        assert len(v) == self.mcoarser+1, \
               'input vector of length %d (should be %d)' \
               % (len(v),self.mcoarser+1)
        y = self.zeros()
        for q in range(self.mcoarser):
            y[2*q] = v[q]
            y[2*q+1] = 0.5 * (v[q] + v[q+1])
        y[-1] = v[-1]
        return y

    def CR(self,v):
        '''Restrict a linear functional (i.e. v in S_k') on the current mesh
        to the next-coarser (k-1) mesh using "canonical restriction".
        Only the interior points are updated.'''
        assert len(v) == self.m+1, \
               'input vector of length %d (should be %d)' % (len(v),self.m+1)
        y = np.zeros(self.mcoarser+1)
        for q in range(1,len(y)-1):
            y[q] = 0.5 * (v[2*q-1] + v[2*q+1]) + v[2*q]
        return y

    def MRO(self,v):
        '''Evaluate the monotone restriction operator on a vector v
        on the current mesh (i.e. v in S_k):
          y = R_k^{k-1} v.
        The result y is on the next-coarser (k-1) mesh, i.e. S_{k-1}.
        See formula (4.22) in G&K(2009).'''
        assert len(v) == self.m+1, \
               'input vector of length %d (should be %d)' % (len(v),self.m+1)
        y = np.zeros(self.mcoarser+1)
        y[0] = max(v[0:2])
        for q in range(1,len(y)-1):
            y[q] = max(v[2*q-1:2*q+2])
        y[-1] = max(v[-2:])
        return y

# mesh hierarchy = [coarse,...,fine]
hierarchy = [None] * (args.j+1)  # list [None,...,None] for indices 0,1,...,j
for k in range(args.j+1):
   hierarchy[k] = MeshLevel(k=k)
mesh = hierarchy[-1]  # fine mesh

# discrete obstacle on fine level and feasible initial iterate
pphi = phi(mesh.xx)
uinitial = np.maximum(pphi,np.zeros(np.shape(mesh.xx)))
uu = uinitial.copy()

# single monotone multigrid V-cycle (unless user just wants pGS)
if args.pgs:
    # sweeps of projected Gauss-Seidel on fine grid
    h2 = mesh.h / 2.0
    hh = mesh.h * mesh.h
    for s in range(args.sweeps):
        for l in range(1,mesh.m):
            # int_0^1 f(x) lambda_p(x) dx \approx h*ff  by midpoint rule
            ff = 0.5 * (fsource(mesh.xx[l]-h2) + fsource(mesh.xx[l]+h2))
            c = 0.5 * (hh*ff + uu[l-1] + uu[l+1]) - uu[l]
            uu[l] += max(c,pphi[l]-uu[l])
else:
    # Algorithm 4.7 in G&K
    # SETUP
    chi = [None] * (args.j+1)      # empty list of length j+1
    chi[args.j] = pphi - uinitial  # fine mesh defect obstacle
    r = mesh.residual(fsource,uu)        # fine mesh residual
    # DOWN-SMOOTH
    for k in range(args.j,0,-1):
        # monotone restriction gives defect obstacle (G&K formulas after (4.22))
        chi[k-1] = hierarchy[k].MRO(chi[k])
        # defect obstacle change on mesh k
        psi = chi[k] - hierarchy[k].prolong(chi[k-1])
        # do projected GS sweeps (FIXME try symmetric)
        hk = hierarchy[k].h
        v = hierarchy[k].zeros()
        print('mesh %d: %d sweeps over %d interior points' \
              % (k,args.sweeps,hierarchy[k].m-1))
        for s in range(args.sweeps):
            for l in range(1,hierarchy[k].m):
                # int_0^1 f(x) lambda_p(x) dx \approx h*ff  by midpoint rule
                xl = hierarchy[k].xx[l]
                c = 0.5 * (hk*hk*r[l] + v[l-1] + v[l+1]) - v[l]
                v[l] += max(c,psi[l])
        hierarchy[k].vstate = v.copy()
        # update the residual and canonically-restrict it
        r += hierarchy[k].residual(fzero,v)
        r = hierarchy[k].CR(r)
    # COARSE SOLVE using fixed number of projected GS sweeps
    coarsesweeps = 1  # FIXME make option
    psi = chi[0]
    h0 = hierarchy[0].h
    v = hierarchy[0].zeros()
    print('coarse: %d sweeps over %d interior points' \
          % (coarsesweeps,hierarchy[0].m-1))
    for s in range(coarsesweeps):
        for l in range(1,hierarchy[0].m):
            xl = hierarchy[0].xx[l]
            c = 0.5 * (h0*h0*r[l] + v[l-1] + v[l+1]) - v[l]
            v[l] += max(c,psi[l])
    hierarchy[0].vstate = v.copy()
    # UP (WITHOUT SMOOTHING)
    for k in range(1,args.j+1):
        print('mesh %d: interpolate up to %d interior points' \
              % (k,hierarchy[k].m-1))
        hierarchy[k].vstate += hierarchy[k].prolong(hierarchy[k-1].vstate)
    # FINALIZE
    uu += mesh.vstate

# evaluate numerical error
udiff = uu - uexact(mesh.xx)
print('  level %d (m = %d) using %d sweeps:  |u-uexact|_2 = %.4e' \
      % (args.j,mesh.m,args.sweeps,mesh.l2norm(udiff)))

# graphical output if desired
if args.show or args.o:
    finalplot(mesh.xx,uinitial,uu)

    #plt.figure(figsize=(15.0,8.0))
    #for k in range(args.j+1):
    #    plt.plot(hierarchy[k].xx,chi[k],'k.-')
    #plt.xlabel('x')

    if args.o:
        plt.savefig(args.o,bbox_inches='tight')
    else:
        plt.show()

