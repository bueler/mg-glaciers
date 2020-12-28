#!/usr/bin/env python3

import numpy as np
import sys, argparse
import matplotlib.pyplot as plt

from meshlevel import MeshLevel1D

parser = argparse.ArgumentParser(description='''
Two level FAS (full approximation storage) scheme for the Liouville-Bratu
problem
  -u'' + nu e^u = 0,  u(0) = u(1) = 0.
FIXME: FULLY describe FAS here so I don't forget it
FIXME: implement MMS
FIXME: make actual V-cycles, i.e. not just two-level
''',add_help=False,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-coarsesweeps', type=int, default=1, metavar='N',
                    help='number of Gauss-Seidel sweeps (default=1)')
parser.add_argument('-cycles', type=int, default=1, metavar='M',
                    help='number of V-cycles (default=2)')
parser.add_argument('-downsweeps', type=int, default=1, metavar='N',
                    help='number of Gauss-Seidel sweeps (default=1)')
parser.add_argument('-fas1help', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-j', type=int, default=2, metavar='J',
                    help='m = 2^{j+1} is number of subintervals in the fine level (default j=2 gives m=8)')
parser.add_argument('-niters', type=int, default=2, metavar='N',
                    help='number of Newton iterations in nonlinear Gauss-Seidel smoother (default=2)')
parser.add_argument('-nu', type=float, default=1.0, metavar='L',
                    help='parameter lambda in Bratu equation (default=1.0)')
parser.add_argument('-show', action='store_true', default=False,
                    help='show plot at end')
parser.add_argument('-upsweeps', type=int, default=0, metavar='N',
                    help='number of Gauss-Seidel sweeps (default=1)')
args, unknown = parser.parse_known_args()
if args.fas1help:
    parser.print_help()
    sys.exit(0)

def residual(mesh,u):
    '''Compute the residual for given u,
       F(u)[v] = int_0^1 u'(x) v'(x) + nu e^{u(x)} v(x) dx
    for v equal to the interior-point hat functions lambda_p at p=1,...,m-1.
    Evaluates first term exactly.  Last two terms are by the trapezoid rule.
    Input mesh is of class MeshLevel.  Input u is a vectors of length m+1.
    The returned vector r is of length m+1 and has r[0]=r[m]=0.'''
    assert len(u) == mesh.m+1, \
           'input vector u is of length %d (should be %d)' % (len(v),mesh.m+1)
    r = mesh.zeros()
    for p in range(1,mesh.m):
        r[p] = (1.0/mesh.h) * (2.0*u[p] - u[p-1] - u[p+1]) \
               + mesh.h * args.nu * np.exp(u[p])
    return r

def ngssweep(mesh,u,frhs):
    '''Do one in-place nonlinear Gauss-Seidel sweep over the interior points
    p=1,...,m-1.  At each point use a fixed number of Newton iterations on
      f(c) = 0
    where
      f(c) = F(u+c lambda_p)[lambda_p] - frhs[lambda_p]
    where v = lambda_p is the pth hat function and F(u)[v] is the same quantity
    as computed by residual().  The integrals are computed by trapezoid rule.
    A Newton step is applied without line search:
      f'(c_k) s_k = - f(c_k)
      c_{k+1} = c_k + s_k.
    '''
    #indices = range(m-1,0,-1) if backward else range(1,m)  [optional backsweep?]
    indices = range(1,mesh.m)
    for p in indices:
        # solve for c:  f(c) = 0
        c = 0
        for n in range(args.niters):
            tmp = mesh.h * args.nu * np.exp(u[p]+c)
            f = (1.0/mesh.h) * (2.0*(u[p]+c) - u[p-1] - u[p+1]) \
                + tmp - mesh.h * frhs[p]
            df = 2.0/mesh.h + tmp
            c -= f / df
        u[p] += c
    return u

# setup meshes
finemesh = MeshLevel1D(k=args.j)
coarsemesh = MeshLevel1D(k=args.j-1)

# FAS V-cycles for two levels, fine and coarse
uu = np.zeros(np.shape(finemesh.xx()))
for s in range(args.cycles):
    # smooth: do args.downsweeps of GS on fine grid using frhs=0
    for q in range(args.downsweeps):
        uu = ngssweep(finemesh,uu,finemesh.zeros())
    # restrict down: compute frhs = R F^h(u^h) + F^H(R u^h)
    rfine = residual(finemesh,uu)
    Ru = finemesh.CR(uu)
    frhs = finemesh.CR(rfine) + residual(coarsemesh,Ru)
    # coarse solve: do args.coarsesweeps of GS on coarse grid
    uuH = coarsemesh.zeros()
    for q in range(args.coarsesweeps):
        uuH = ngssweep(coarsemesh,uuH,frhs)
    # prolong up
    uu += finemesh.prolong(uuH - Ru)
    # smooth: do args.upsweeps of GS on fine grid using frhs=0
    for q in range(args.upsweeps):
        uu = ngssweep(finemesh,uu,finemesh.zeros())

# report on computation including numerical error
print('  m=%d mesh: |u|_2=%.6f' % (finemesh.m,finemesh.l2norm(uu)))

# graphical output if desired
if args.show:
    import matplotlib
    font = {'size' : 20}
    matplotlib.rc('font', **font)
    lines = {'linewidth': 2}
    matplotlib.rc('lines', **lines)

    plt.figure(figsize=(15.0,8.0))
    plt.plot(finemesh.xx(),uu,'k',linewidth=4.0)
    plt.xlabel('x')

    plt.show()
