#!/usr/bin/env python3

# TODO:
#   count work units
#   implement F-cycle
#   implement quasilinear example like p-laplacian

import numpy as np
import sys, argparse

from meshlevel import MeshLevel1D

parser = argparse.ArgumentParser(description='''
FAS (full approximation storage) scheme for the nonlinear Liouville-Bratu
problem
  -u'' - lambda e^u = g,  u(0) = u(1) = 0
where lambda is constant (adjust with -lam) and g(x) is given.  The
default case has g(x)=0.  In the -mms case g(x) is computed so that
u(x) = sin(3 pi x) is the exact solution.

Solution is by piecewise-linear finite elements on a fine mesh with
m = 2^{j+1} subintervals and m-1 nodes.  Note -j sets the fine mesh.  To
set up the FAS multigrid solver we create -levels levels of meshes,
defaulting to j+1 levels.  The hierarchy is meshes[k] for k=0,1,...,j,
listed from coarse to fine.

The solver uses nonlinear Gauss-Seidel (NGS), which uses -niters scalar
Newton iterations, as a smoother.  Note NGS is also the coarse mesh solver.

Only V-cycles are implemented; set the number of cycles with -cycles.
Monitor the residual between V-cycles with -monitor.  Set the number of
down- and up-smoother NGS sweeps (-downsweeps,-upsweeps) and coarsest-mesh
NGS sweeps (-coarsesweeps).  One can revert to NGS sweeps only on the fine
mesh (-ngsonly), but then set -downsweeps to a large value.

Show the solution in Matplotlib graphics with -show.  For more information
on runtime options use -fas1help.  For documentation see ../doc/fas.pdf.
''',add_help=False,formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-coarsesweeps', type=int, default=1, metavar='N',
                    help='NGS sweeps on coarsest mesh (default=1)')
parser.add_argument('-cycles', type=int, default=1, metavar='M',
                    help='number of V-cycles (default=1)')
parser.add_argument('-downsweeps', type=int, default=1, metavar='N',
                    help='NGS sweeps before coarse-mesh correction (default=1)')
parser.add_argument('-fas1help', action='store_true', default=False,
                    help='print help for this program and quit')
parser.add_argument('-j', type=int, default=2, metavar='J',
                    help='m=2^{j+1} intervals in fine mesh (default j=2, m=8)')
parser.add_argument('-lam', type=float, default=1.0, metavar='L',
                    help='parameter lambda in Bratu equation (default=1.0)')
parser.add_argument('-levels', type=int, default=-1, metavar='J',
                    help='number of levels in V-cycle (default: levels=j+1)')
parser.add_argument('-mms', action='store_true', default=False,
                    help='manufactured problem with known exact solution')
parser.add_argument('-monitor', action='store_true', default=False,
                    help='print residual norms')
parser.add_argument('-monitorcoarseupdate', action='store_true', default=False,
                    help='print norms for the coarse-mesh update vector')
parser.add_argument('-ngsonly', action='store_true', default=False,
                    help='only do -downsweeps NGS sweeps at each iteration')
parser.add_argument('-niters', type=int, default=2, metavar='N',
                    help='Newton iterations in NGS smoothers (default=2)')
parser.add_argument('-show', action='store_true', default=False,
                    help='show plot at end')
parser.add_argument('-upsweeps', type=int, default=1, metavar='N',
                    help='NGS sweeps after coarse-mesh correction (default=1)')
args, unknown = parser.parse_known_args()

if args.fas1help:
    parser.print_help()
    sys.exit(0)
if args.levels < 1:
    args.levels = args.j+1

def mmsevaluate(x):
    u = np.sin(3.0 * np.pi * x)
    g = 9.0 * np.pi**2 * u - args.lam * np.exp(u)
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
               - mesh.h * args.lam * np.exp(u[p])
    return F

def ngssweep(mesh,w,ell,forward=True):
    '''Do one in-place nonlinear Gauss-Seidel sweep over the interior points
    p=1,...,m-1.  At each point use a fixed number of Newton iterations on
      f(c) = 0
    where
      f(c) = r(w+c lambda_p)[lambda_p]
    where v = lambda_p is the pth hat function and
      r(w)[v] = ell[v] - F(w)[v]
    is the residual for w.  The implied integrals in ell[.] and F(w)[.] are
    computed by the trapezoid rule.  A Newton step is computed without line
    search:
      f'(c_k) s_k = - f(c_k),   c_{k+1} = c_k + s_k.
    '''
    if forward:
        indices = range(1,mesh.m)
    else:
        indices = range(mesh.m-1,0,-1)
    for p in indices:
        c = 0   # because previous iterate u is close to correct
        for n in range(args.niters):
            tmp = mesh.h * args.lam * np.exp(w[p]+c)
            f = - (1.0/mesh.h) * (2.0*(w[p]+c) - w[p-1] - w[p+1]) \
                + tmp + ell[p]
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
def vcycle(k,u,ell):
    uin = u.copy()
    if k == kcoarse:
        # coarse solve: NGS sweeps on coarse mesh
        for q in range(args.coarsesweeps):
            ngssweep(meshes[k],u,ell)
        return u, u - uin
    else:
        assert k > kcoarse
        # smooth: NGS sweeps on fine mesh
        for q in range(args.downsweeps):
            ngssweep(meshes[k],u,ell)
        # restrict down: compute ell = R' (f^h - F^h(u^h)) + F^{2h}(R u^h)
        rfine = ell - FF(meshes[k],u)  # residual on the fine mesh
        Ru = meshes[k].Rfw(u)
        coarseell = meshes[k].CR(rfine) + FF(meshes[k-1],Ru)
        # recurse
        _, ducoarse = vcycle(k-1,Ru,coarseell)
        if args.monitorcoarseupdate:
            print('     ' + '  ' * (args.j + 1 - k), end='')
            print('coarse update norm %.5e' % meshes[k-1].l2norm(ducoarse))
        # prolong up
        u += meshes[k].prolong(ducoarse)
        # smooth: NGS sweeps on fine mesh
        for q in range(args.upsweeps):
            ngssweep(meshes[k],u,ell,forward=False)
        return u, u - uin

# compute fine mesh right-hand side, a linear functional
ellg = meshes[args.j].zeros()
if args.mms:
    _, g = mmsevaluate(meshes[args.j].xx())
    g *= meshes[args.j].h
    ellg[1:-1] = g[1:-1]

def printresidualnorm(s,mesh,u,ell):
    rnorm = mesh.l2norm(ell - FF(mesh,u))
    print('  %d: residual norm %.5e' % (s,rnorm))

# SOLVE:  do V-cycles or NGS sweeps, with residual monitoring
uu = meshes[args.j].zeros()
for s in range(args.cycles):
    if args.monitor:
        printresidualnorm(s,meshes[args.j],uu,ellg)
    if args.ngsonly:
        for q in range(args.downsweeps):
            ngssweep(meshes[args.j],uu,ellg)
        continue
    else:
        uu, _ = vcycle(args.j,uu,ellg)
if args.monitor:
    printresidualnorm(args.cycles,meshes[args.j],uu,ellg)

# report on computation
if args.ngsonly:
    print('  m=%d mesh using %d sweeps of NGS only: |u|_2=%.6f' \
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
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    font = {'size' : 20}
    mpl.rc('font', **font)
    lines = {'linewidth': 2}
    mpl.rc('lines', **lines)
    plt.figure(figsize=(15.0,8.0))
    plt.plot(meshes[args.j].xx(),uu,'k',linewidth=4.0,label='numerical solution')
    if args.mms:
        plt.plot(meshes[args.j].xx(),uexact,'k--',linewidth=4.0,label='exact solution')
        plt.legend()
    plt.xlabel('x')
    plt.show()

