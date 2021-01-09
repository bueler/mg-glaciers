#!/usr/bin/env python3

# TODO:
#   implement F-cycle
#   implement quasilinear example like p-laplacian

import numpy as np
import sys, argparse

from meshlevel import MeshLevel1D
from problems import LiouvilleBratu1D
from cycles import FAS

prs = argparse.ArgumentParser(description='''
FAS (full approximation storage) scheme for the nonlinear ordinary
differential equation boundary value problems.  The default problem
is Liouville-Bratu:
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
down- and up-smoother NGS sweeps (-down,-up) and coarsest-mesh NGS sweeps
(-coarse).  One can revert to only using NGS sweeps on the fine mesh
(-ngsonly), but the user should set -down to a large value in that case.

Show the solution in Matplotlib graphics with -show.  For more information
on runtime options use -fas1help.  For documentation see ../doc/fas.pdf.
''',add_help=False,formatter_class=argparse.RawTextHelpFormatter)
prs.add_argument('-coarse', type=int, default=1, metavar='N',
                 help='number of NGS sweeps on coarsest mesh (default=1)')
prs.add_argument('-cycles', type=int, default=1, metavar='M',
                 help='number of V-cycles (default=1)')
prs.add_argument('-down', type=int, default=1, metavar='N',
                 help='number of NGS sweeps before coarse correction (default=1)')
prs.add_argument('-fas1help', action='store_true', default=False,
                 help='print help for this program and quit')
prs.add_argument('-j', type=int, default=2, metavar='J',
                 help='m=2^{j+1} intervals in fine mesh (default j=2, m=8)')
prs.add_argument('-lam', type=float, default=1.0, metavar='L',
                 help='parameter lambda in Bratu equation (default=1.0)')
prs.add_argument('-levels', type=int, default=-1, metavar='J',
                 help='number of levels in V-cycle (default: levels=j+1)')
prs.add_argument('-mms', action='store_true', default=False,
                 help='manufactured problem with known exact solution')
prs.add_argument('-monitor', action='store_true', default=False,
                 help='print residual norms')
prs.add_argument('-monitorcoarseupdate', action='store_true', default=False,
                 help='print norms for the coarse-mesh update vector')
prs.add_argument('-ngsonly', action='store_true', default=False,
                 help='only do -downsweeps NGS sweeps at each iteration')
prs.add_argument('-niters', type=int, default=2, metavar='N',
                 help='Newton iterations in NGS smoothers (default=2)')
prs.add_argument('-show', action='store_true', default=False,
                 help='show plot at end')
prs.add_argument('-up', type=int, default=1, metavar='N',
                 help='number of NGS sweeps after coarse correction (default=1)')
args, unknown = prs.parse_known_args()

# provide usage help
if unknown:
    print('ERROR: unknown arguments ... try -fas1help for usage')
    sys.exit(1)
if args.fas1help:
    prs.print_help()
    sys.exit(0)

# setup mesh hierarchy
if args.levels < 1:
    args.levels = args.j+1
meshes = [None] * (args.j + 1)     # spots for k=0,...,j meshes
assert (args.levels >= 1) and (args.levels <= args.j + 1)
kcoarse = args.j-args.levels+1
for k in range(kcoarse,args.j+1):  # create meshes for the ones we use
    meshes[k] = MeshLevel1D(k=k)

# initialize problem
prob = LiouvilleBratu1D(lam=args.lam)

# initialize FAS and its parameters
fas = FAS(meshes,prob,kcoarse=kcoarse,kfine=args.j,
          mms=args.mms,coarse=args.coarse,down=args.down,up=args.up,
          niters=args.niters,monitorupdate=args.monitorcoarseupdate)

def printresidualnorm(s,w,ell):
    if args.monitor:
        print('  %d: residual norm %.5e' % (s,fas.residualnorm(w,ell)))

# SOLVE:  do V-cycles or NGS sweeps, with residual monitoring
uu = meshes[args.j].zeros()
ellg = fas.rhs(args.j)
for s in range(args.cycles):
    printresidualnorm(s,uu,ellg)
    if args.ngsonly:
        for q in range(args.down):
            prob.ngssweep(meshes[args.j],uu,ellg,niters=args.niters)
        fas.wu[args.j] += args.down  # add count into FAS work units array
    else:
        fas.vcycle(args.j,uu,ellg)
printresidualnorm(args.cycles,uu,ellg)

# report on computation
if args.ngsonly:
    print('  m=%d mesh using %d sweeps of NGS only (%d work units): |u|_2=%.6f' \
          % (meshes[args.j].m,args.cycles*args.down,
             fas.wutotal(),meshes[args.j].l2norm(uu)))
else:
    print('  m=%d mesh using %d V(%d,%d,%d) cycles (%d work units): |u|_2=%.6f' \
          % (meshes[args.j].m,args.cycles,args.down,args.coarse,args.up,
             fas.wutotal(),meshes[args.j].l2norm(uu)))
if args.mms:
    uexact, _ = prob.mms(meshes[args.j].xx())
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
    plt.plot(meshes[args.j].xx(),uu,'k',linewidth=4.0,
             label='numerical solution')
    if args.mms:
        plt.plot(meshes[args.j].xx(),uexact,'k--',linewidth=4.0,
                 label='exact solution')
        plt.legend()
    plt.xlabel('x')
    plt.show()

