#!/usr/bin/env python3

import numpy as np
import sys, argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='''
Solve a 1D obstacle problem:            /1
                                min     |  1/2 (u')^2 - f u
                              u >= phi  /0
where  phi(x) = 8x(1-x)-1,  f(x) = -2,  and  u in H_0^1[0,1].
(Thus where u>phi we have -u''=-2.)

FIXME this version is projected Gauss-Seidel

Reference: Gräser, C., & Kornhuber, R. (2009). Multigrid methods for
obstacle problems. J. Comput. Math., 1-44.
''',add_help=False)
parser.add_argument('-j', type=int, default=3, metavar='J',
                    help='fine grid level (default j=3 gives 8 subintervals)')
parser.add_argument('-obstacle1help', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-o', metavar='FILE.pdf', type=str, default='',
                    help='save plot at end in image file')
parser.add_argument('-show', action='store_true', default=False,
                    help='show plot at end')
parser.add_argument('-sweeps', type=int, default=3, metavar='N',
                    help='number of sweeps of projected Gauss-Seidel (default=3)')
args, unknown = parser.parse_known_args()
if args.obstacle1help:
    parser.print_help()
    sys.exit(0)

def phi(x):
    '''The obstacle:  u >= phi.'''
    return 8.0 * x * (1.0 - x) - 1.0

def f(x):
    '''The source term:  -u'' = f.'''
    return - 2.0 * np.ones(np.shape(x))

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
    plt.plot(xx,uinitial,'k--',label='initial',linewidth=2.0)
    plt.plot(xx,ufinal,'k',label='final',linewidth=4.0)
    plt.plot(xx,phi(xx),'r',label='obstacle',linewidth=2.0)
    plt.plot(xx,uexact(xx),'g',label='final',linewidth=2.0)
    plt.axis([0.0,1.0,-0.3,1.1])
    plt.legend(fontsize=20.0)
    plt.xlabel('x')

class Level(object):
    '''Encapsulate a mesh level.  Level(k=0) is coarse and Level(k=j) is fine.'''

    def __init__(self, k=None):
        self.k = k
        self.m = 2**self.k
        self.h = 1.0 / self.m
        self.xx = np.linspace(0.0,1.0,self.m+1)  # indices 0,1,...,m
        self.pp = self.xx[1:-1]                  # interior nodes 1,...,m-1

    def l2norm(self, u):
        '''L^2[0,1] norm computed with trapezoid rule.'''
        return np.sqrt(self.h * (0.5*u[0]*u[0] + np.sum(u[1:-1]*u[1:-1]) + 0.5*u[-1]*u[-1]))

    def hat(self,p):
        u = np.zeros(self.m+1)
        u[p] = 1.0
        return u

    def MRO(self,v):
        '''Evaluate the monotone restriction operator on v:
          y = R_k^{k-1} v.
        The result y is on the next coarser (k-1) mesh.  See
        formula (4.22) in G&K(2009).'''
        assert len(v) == self.m+1, \
               'input vector of length %d (should be %d)' % (len(v),self.m+1)
        y = np.zeros(2**(self.k-1)+1)
        y[0] = max(v[0:2])
        for q in range(1,len(y)-1):
            y[q] = max(v[2*q-1:2*q+2])
        y[-1] = max(v[-2:])
        return y

# mesh and discrete problem
mesh = Level(k=args.j)
ff = f(mesh.xx)
pphi = phi(mesh.xx)

# initial iterate is feasible
uinitial = np.maximum(pphi,np.zeros(np.shape(mesh.xx)))
uu = uinitial.copy()

# sweeps of projected GS
for s in range(args.sweeps):
    for l in range(1,mesh.m):
        c = 0.5 * (mesh.h*mesh.h*ff[l] + uu[l-1] + uu[l+1]) - uu[l]
        uu[l] += max(c,pphi[l]-uu[l])

# evaluate numerical error
udiff = uu - uexact(mesh.xx)
print('  level %d (m = %d) using %d sweeps:  |u-uexact|_2 = %.4e' \
      % (args.j,mesh.m,args.sweeps,mesh.l2norm(udiff)))

# graphical output if desired
if args.show or args.o:
    finalplot(mesh.xx,uinitial,uu)
    #plt.figure(figsize=(15.0,8.0))
    #plt.plot(mesh.xx,pphi-uu,'k.-')
    #plt.plot(mesh.xx[::2],mesh.MRO(pphi-uu),'k.--')
    #plt.xlabel('x')
    if args.o:
        plt.savefig(args.o,bbox_inches='tight')
    else:
        plt.show()

