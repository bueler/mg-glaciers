#!/usr/bin/env python3

# FIXME count work units

# convergence in -mms case by brutal NGS sweeps:
#$ for JJ in 1 2 3 4 5 6; do ./fas1.py -downsweeps 10000 -ngsonly -mms -j $JJ; done
#  m=4 mesh using 10000 sweeps of NGS: |u|_2=1.165660
#  numerical error: |u-u_exact|_2=4.6169e-01
#  m=8 mesh using 10000 sweeps of NGS: |u|_2=0.797087
#  numerical error: |u-u_exact|_2=9.0323e-02
#  m=16 mesh using 10000 sweeps of NGS: |u|_2=0.728361
#  numerical error: |u-u_exact|_2=2.1331e-02
#  m=32 mesh using 10000 sweeps of NGS: |u|_2=0.712347
#  numerical error: |u-u_exact|_2=5.2591e-03
#  m=64 mesh using 10000 sweeps of NGS: |u|_2=0.708412
#  numerical error: |u-u_exact|_2=1.3102e-03
#  m=128 mesh using 10000 sweeps of NGS: |u|_2=0.707433
#  numerical error: |u-u_exact|_2=3.2592e-04

# FIXME these runs show there is a scaling problem
# $ for JJ in 2 3 4 5; do ./fas1.py -mms -cycles 3 -j $JJ -monitor -show; done

# 5 cycles of 2-level with very accurate coarse-mesh solve:  FAILS
#   $ for JJ in 1 2 3 4 5 6; do ./fas1.py -j $JJ -cycles 5 -coarsesweeps 1000 -mms -levels 2; done

import numpy as np
import sys, argparse
import matplotlib.pyplot as plt

import matplotlib as mpl
font = {'size' : 20}
mpl.rc('font', **font)
lines = {'linewidth': 2}
mpl.rc('lines', **lines)

from meshlevel import MeshLevel1D

parser = argparse.ArgumentParser(description='''
FAS (full approximation storage) scheme for the nonlinear Liouville-Bratu
problem
  -u'' - mu e^u = 0,  u(0) = u(1) = 0
where nu is constant (adjust with -nu).  In the -mms case the equation is
  -u'' - mu e^u = g,  u(0) = u(1) = 0
for a given function g computed so that u(x) = sin(3 pi x) is the exact
solution.

Solution is by piecewise-linear finite elements on a fine mesh with
m = 2^{j+1} subintervals, where -j sets the fine mesh, and m-1 nodes.  To
set up the FAS multigrid solver we create -levels levels of meshes,
defaulting to j+1 levels meshes[k] for k=0,1,...,j (from coarse to fine).
The solver uses nonlinear Gauss-Seidel (NGS; uses -niters Newton iterations)
as a smoother, and NGS is also the coarse mesh solver.  Only V-cycles are
implemented; set the number of cycles with -cycles.  One can set the number
of down- and up-smoother NGS sweeps (-downsweeps,-upsweeps) and the number
of coarsest-mesh NGS sweeps (-coarsesweeps).  One can also revert to NGS
sweeps only on the fine mesh (-ngsonly; set -downsweeps to a large value).
Monitor the V-cycles with -monitor and show the solution with -show.

For more information on runtime options use -fas1help.  For documentation
see the PDF mg-glaciers/fas/fas.pdf.
''',add_help=False,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-coarsesweeps', type=int, default=5, metavar='N',
                    help='nonlinear Gauss-Seidel sweeps (default=5)')
parser.add_argument('-cycles', type=int, default=1, metavar='M',
                    help='number of V-cycles (default=1)')
parser.add_argument('-downsweeps', type=int, default=1, metavar='N',
                    help='nonlinear Gauss-Seidel sweeps (default=1)')
parser.add_argument('-fas1help', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-j', type=int, default=2, metavar='J',
                    help='m=2^{j+1} intervals in fine mesh (default j=2, m=8)')
parser.add_argument('-levels', type=int, default=-1, metavar='J',
                    help='number of levels in V-cycle (default: levels=j+1)')
parser.add_argument('-mms', action='store_true', default=False,
                    help='manufactured problem with known exact solution')
parser.add_argument('-monitor', action='store_true', default=False,
                    help='print residual and update norms')
parser.add_argument('-mu', type=float, default=1.0, metavar='L',
                    help='parameter lambda in Bratu equation (default=1.0)')
parser.add_argument('-ngsonly', action='store_true', default=False,
                    help='only do -downsweeps NGS sweeps at each iteration')
parser.add_argument('-niters', type=int, default=2, metavar='N',
                    help='Newton iterations in NGS smoothers (default=2)')
parser.add_argument('-show', action='store_true', default=False,
                    help='show plot at end')
parser.add_argument('-upsweeps', type=int, default=1, metavar='N',
                    help='nonlinear Gauss-Seidel sweeps (default=1)')
args, unknown = parser.parse_known_args()
if args.fas1help:
    parser.print_help()
    sys.exit(0)
if args.levels < 1:
    args.levels = args.j+1

def mmsevaluate(x):
    u = np.sin(3.0 * np.pi * x)
    g = 9.0 * np.pi**2 * u - args.mu * np.exp(u)
    return u, g

def FF(mesh,u):
    '''Compute the nonlinear operator for given u,
       F(u)[v] = int_0^1 u'(x) v'(x) + nu e^{u(x)} v(x) dx
    for v equal to the interior-point hat functions lambda_p at p=1,...,m-1.
    Evaluates first term exactly.  Last term is by the trapezoid rule.
    Input mesh is of class MeshLevel.  Input u is a vectors of length m+1.
    The returned vector F is of length m+1 and has F[0]=F[m]=0.'''
    assert len(u) == mesh.m+1, \
           'input vector u is of length %d (should be %d)' % (len(u),mesh.m+1)
    F = mesh.zeros()
    for p in range(1,mesh.m):
        F[p] = (1.0/mesh.h) * (2.0*u[p] - u[p-1] - u[p+1]) \
               - mesh.h * args.mu * np.exp(u[p])
    return F

def residual(mesh,w,frhs):
    '''Compute the residual of "F(u)=f" for given u, namely
       r(w)[v] = int_0^1 f(x) v(x) dx - F(w)[v]
    for v equal to the interior-point hat functions lambda_p at p=1,...,m-1.'''
    assert len(w) == mesh.m+1, \
           'input vector w is of length %d (should be %d)' % (len(w),mesh.m+1)
    return mesh.h * frhs - FF(mesh,w)

def ngssweep(mesh,w,frhs,forward=True):
    '''Do one in-place nonlinear Gauss-Seidel sweep over the interior points
    p=1,...,m-1.  At each point use a fixed number of Newton iterations on
      f(c) = 0
    where
      f(c) = r(w+c lambda_p)[lambda_p]
    where v = lambda_p is the pth hat function and r(w)[v] is computed by
    residual().  The integrals are computed by trapezoid rule.  A Newton step
    is applied without line search:
      f'(c_k) s_k = - f(c_k),   c_{k+1} = c_k + s_k.
    '''
    if forward:
        indices = range(1,mesh.m)
    else:
        indices = range(mesh.m-1,0,-1)
    for p in indices:
        c = 0   # because previous iterate u is close to correct
        for n in range(args.niters):
            tmp = mesh.h * args.mu * np.exp(w[p]+c)
            f = - (1.0/mesh.h) * (2.0*(w[p]+c) - w[p-1] - w[p+1]) \
                + tmp + mesh.h * frhs[p]
            df = - 2.0/mesh.h + tmp
            c -= f / df
        w[p] += c
    return None

# setup mesh hierarchy
meshes = [None] * (args.j + 1)     # spots for k=0,...,j meshes
assert (args.levels >= 1) and (args.levels <= args.j + 1)
kcoarse = args.j-args.levels+1
for k in range(kcoarse,args.j+1):  # create meshes for the ones we use
    meshes[k] = MeshLevel1D(k=k)

# FAS V-cycle for levels k down to k=kcoarse
def vcycle(k,u,frhs):
    uin = u.copy()
    if k == kcoarse:
        # coarse solve: NGS sweeps on coarse mesh
        for q in range(args.coarsesweeps):
            ngssweep(meshes[k],u,frhs)
        return u, u - uin
    else:
        assert k > kcoarse
        # smooth: NGS sweeps on fine mesh
        for q in range(args.downsweeps):
            ngssweep(meshes[k],u,frhs)
        # restrict down: compute frhs = R (f^h - F^h(u^h)) + F^H(R u^h)
        rfine = residual(meshes[k],u,frhs)
        Ru = meshes[k].VR0(u)
        coarsefrhs = meshes[k].VR(rfine) + FF(meshes[k-1],Ru)
        # recurse
        _, ducoarse = vcycle(k-1,Ru,coarsefrhs)
        if args.monitor:
            print('     ' + '  ' * (args.j + 1 - k), end='')
            print('coarse update norm %.5e' % meshes[k-1].l2norm(ducoarse))
        # prolong up
        u += meshes[k].prolong(ducoarse)
        # smooth: NGS sweeps on fine mesh
        for q in range(args.upsweeps):
            ngssweep(meshes[k],u,frhs,forward=False)
        return u, u - uin

# compute fine mesh right-hand side FIXME this is point wise function?
if args.mms:
    _, f = mmsevaluate(meshes[args.j].xx())
    f[0] = 0.0
    f[meshes[args.j].m] = 0.0
else:
    f = meshes[args.j].zeros()

# SOLVE:  do V-cycles or NGS sweeps, with residual monitoring
uu = meshes[args.j].zeros()
for s in range(args.cycles):
    if args.monitor:
        rnorm = meshes[args.j].l2norm(residual(meshes[args.j],uu,f))
        print('  %d: residual norm %.5e' % (s,rnorm))
    if args.ngsonly:
        for q in range(args.downsweeps):
            ngssweep(meshes[args.j],uu,f)
        continue
    else:
        uu, _ = vcycle(args.j,uu,f)
if args.monitor:
    rnorm = meshes[args.j].l2norm(residual(meshes[args.j],uu,f))
    print('  %d: residual norm %.5e' % (args.cycles,rnorm))

# report on computation
if args.ngsonly:
    print('  m=%d mesh using %d sweeps of NGS: |u|_2=%.6f' \
          % (meshes[args.j].m,args.cycles*args.downsweeps,
             meshes[args.j].l2norm(uu)))
else:
    print('  m=%d mesh using %d V(%d,%d,%d) cycles: |u|_2=%.6f' \
          % (meshes[args.j].m,args.cycles,
             args.downsweeps,args.coarsesweeps,args.upsweeps,
             meshes[args.j].l2norm(uu)))
if args.mms:
    uexact, _ = mmsevaluate(meshes[args.j].xx())
    print('  numerical error: |u-u_exact|_2=%.4e' \
          % (meshes[args.j].l2norm(uu - uexact)))

# graphical output if desired
if args.show:
    plt.figure(figsize=(15.0,8.0))
    plt.plot(meshes[args.j].xx(),uu,'k',linewidth=4.0,label='numerical solution')
    if args.mms:
        plt.plot(meshes[args.j].xx(),uexact,'k--',linewidth=4.0,label='exact solution')
        plt.legend()
    plt.xlabel('x')
    plt.show()

