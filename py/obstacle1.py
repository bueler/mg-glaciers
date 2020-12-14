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
''',add_help=False)
parser.add_argument('-m', type=int, default=8, metavar='N',
                    help='fine grid has m subintervals (default=8)')
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

# mesh and discrete problem
xx = np.linspace(0.0,1.0,args.m+1)  # indices 0,1,...,m
h = 1.0 / args.m
ff = f(xx)
pphi = phi(xx)

# initial iterate is feasible
uinitial = np.maximum(pphi,np.zeros(args.m+1))
uu = uinitial.copy()

# sweeps of projected GS
for s in range(args.sweeps):
    for l in range(1,args.m):
        c = 0.5 * (h*h*ff[l] + uu[l-1] + uu[l+1]) - uu[l]
        uu[l] += max(c,pphi[l]-uu[l])

# evaluate numerical error
udiff = uu - uexact(xx)
l2 = np.sqrt(np.sum(udiff*udiff))
print('  m = %4d from %4d sweeps:  |u-uexact|_2 = %.4e' % (args.m,args.sweeps,l2))

# graphical output if desired
if args.show or args.o:
    finalplot(xx,uinitial,uu)
    if args.o:
        plt.savefig(args.o,bbox_inches='tight')
    else:
        plt.show()

