#!/usr/bin/env python3

import numpy as np
import sys, argparse
import matplotlib.pyplot as plt

from meshlevel import MeshLevel1D

parser = argparse.ArgumentParser(description='''
Two level FAS (full approximation storage) scheme for the nonlinear
(semilinear) Liouville-Bratu problem
  -u'' + nu e^u = 0,  u(0) = u(1) = 0
where nu is constant.  Let F be the residual associated to the weak form,
  F(u)[v] = int_0^1 u'(x) v'(x) + nu e^{u(x)} v(x) dx,
acting on u and v in H_0^1[0,1].

The fine mesh which has m intervals and p=1,...,m-1 interior points.  On this
fine mesh the residual is denoted F^h, and on the coarse mesh it is F^H.
These functions act on piecewise-linear and continuous functions u in
vector spaces S^h,S^H respectively.  These spaces have hat functions
{lambda_p(x)} as a basis.  (Function residual() below computes F(w) on the
given mesh for a given iterate w.  The point values are F(w)[lambda_p].)

For the unknown, exact fine mesh solution
  u^h(x) = sum_{p=1}^{m-1} u^p lambda_p(x)
we want to solve
  F^h(u^h)[v] = 0
for all hat functions v(x) = lambda_p(x).

Suppose w^h is an iterate on the fine mesh.  The smoother is nonlinear
Gauss-Seidel, which updates w^h.  (Function ngssweep() below computes one sweep
by using a fixed number of scalar Newton iterations at each point.)  Sweeps of
this method accomplish the following on the fine mesh:
  1. making the residual F^h(w^h) smooth, but not small, and
  2. making the difference u^h - w^h smooth, but not small.

Noting that F is nonlinear in u, the FAS method now proposes a new equation
on the coarse mesh.  If the fine-mesh solver has already been applied then
the new equation relates smooth quantities which should be well-approximated
on the coarse mesh,
  F^H(u^H) - F^H(R w^h) = R F^h(w^h),
where u^H is the exact solution of this equation on the coarse mesh.  Here R
is the restriction of a vector on the fine mesh to the coarse mesh.  (Note
that MeshLevel1D.CR() computes this "canonical restriction" by "full-weighting",
i.e. by averaging onto the coarse mesh.)  Note that if w^h were the exact
solution to the fine mesh problem then the right side of this coarse mesh
equation would be zero and the solution would be u^H = R w^h by well-posedness.

Thus on the coarse mesh we need to solve
  F^H(u^H)[v] = f[v],
for v a hat function on the coarse mesh, where
  f[v] = R F^h(w^h)[v] + F^H(R w^h)[v]
We do this by (perhaps many) sweeps of nonlinear Gauss-Seidel.  (Note that
ngssweep() below allows a right-hand side function (vector) f.)  After the
coarse mesh solution we have u^H (assumed exact for presentation).  The final
step of two-grid FAS is the update
  w^h <-- w^h + P(u^H - R w^h).
Here P is prolongation, acting by linear interpolation.  Note the update is
zero if w^h is already the exact fine mesh solution.

FIXME: implement MMS
FIXME: make actual V-cycles, i.e. not just two-level
FIXME: count work units
''',add_help=False,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-coarsesweeps', type=int, default=5, metavar='N',
                    help='number of nonlinear Gauss-Seidel sweeps (default=5)')
parser.add_argument('-cycles', type=int, default=1, metavar='M',
                    help='number of V-cycles (default=1)')
parser.add_argument('-downsweeps', type=int, default=1, metavar='N',
                    help='number of nonlinear Gauss-Seidel sweeps (default=1)')
parser.add_argument('-fas1help', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-j', type=int, default=2, metavar='J',
                    help='m = 2^{j+1} is subintervals in fine mesh (default j=2 gives m=8)')
parser.add_argument('-monitor', action='store_true', default=False,
                    help='print residual and update norms')
parser.add_argument('-ngsonly', action='store_true', default=False,
                    help='only do -downsweeps NGS sweeps at each iteration')
parser.add_argument('-niters', type=int, default=2, metavar='N',
                    help='number of Newton iterations in NGS smoothers (default=2)')
parser.add_argument('-nu', type=float, default=1.0, metavar='L',
                    help='parameter lambda in Bratu equation (default=1.0)')
parser.add_argument('-show', action='store_true', default=False,
                    help='show plot at end')
parser.add_argument('-upsweeps', type=int, default=1, metavar='N',
                    help='number of nonlinear Gauss-Seidel sweeps (default=1)')
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

def ngssweep(mesh,u,frhs,forward=True):
    '''Do one in-place nonlinear Gauss-Seidel sweep over the interior points
    p=1,...,m-1.  At each point use a fixed number of Newton iterations on
      f(c) = 0
    where
      f(c) = F(u+c lambda_p)[lambda_p] - frhs[lambda_p]
    where v = lambda_p is the pth hat function and F(u)[v] is computed by
    residual().  The integrals are computed by trapezoid rule.  A Newton step
    is applied without line search:
      f'(c_k) s_k = - f(c_k)
      c_{k+1} = c_k + s_k.
    '''
    if forward:
        indices = range(1,mesh.m)
    else:
        indices = range(mesh.m-1,0,-1)
    for p in indices:
        c = 0   # because previous iterate u is close to correct
        for n in range(args.niters):
            tmp = mesh.h * args.nu * np.exp(u[p]+c)
            f = (1.0/mesh.h) * (2.0*(u[p]+c) - u[p-1] - u[p+1]) \
                + tmp - mesh.h * frhs[p]
            df = 2.0/mesh.h + tmp
            c -= f / df
        u[p] += c
    return None

# setup meshes
finemesh = MeshLevel1D(k=args.j)
coarsemesh = MeshLevel1D(k=args.j-1)

# FAS V-cycles for two levels, fine and coarse
uu = np.zeros(np.shape(finemesh.xx()))
for s in range(args.cycles):
    if args.monitor:
        print('  %d: residual norm %.5e' \
              % (s,finemesh.l2norm(residual(finemesh,uu))))
    # smooth: do args.downsweeps of NGS on fine mesh
    for q in range(args.downsweeps):
        ngssweep(finemesh,uu,finemesh.zeros())
    if args.ngsonly:
        continue
    # restrict down: compute frhs = R F^h(u^h) + F^H(R u^h)
    Ffine = residual(finemesh,uu)
    Ru = finemesh.VR(uu)
    frhs = finemesh.VR(Ffine) + residual(coarsemesh,Ru)
    # coarse solve: do args.coarsesweeps of NGS on coarse mesh
    uuH = Ru.copy()
    for q in range(args.coarsesweeps):
        ngssweep(coarsemesh,uuH,frhs)
    # prolong up
    if args.monitor:
        print('    coarse update norm %.5e' \
              % coarsemesh.l2norm(uuH - Ru))
    uu += finemesh.prolong(uuH - Ru)
    # smooth: do args.upsweeps of NGS on fine mesh
    for q in range(args.upsweeps):
        ngssweep(finemesh,uu,finemesh.zeros(),forward=False)

if args.monitor:
    print('  %d: residual norm %.5e' \
          % (s+1,finemesh.l2norm(residual(finemesh,uu))))

# report on computation
if args.ngsonly:
    print('  m=%d mesh using %d sweeps of NGS: |u|_2=%.6f' \
          % (finemesh.m,args.cycles*args.downsweeps,finemesh.l2norm(uu)))
else:
    print('  m=%d mesh using %d V(%d,%d,%d) cycles: |u|_2=%.6f' \
          % (finemesh.m,args.cycles,
             args.downsweeps,args.coarsesweeps,args.upsweeps,
             finemesh.l2norm(uu)))

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
