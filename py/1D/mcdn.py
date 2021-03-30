'''Module implementing the nonlinear full approximation storage (FAS) extension
of the multilevel constraint decomposition (MCD) method for the shallow ice
approximation (SIA) obstacle problem.'''

# TODO:
#   * implement p=4 p-Laplacian example to see how much better it does
#   * implement F-cycles
#   * make -down 0 -up 1 the default everywhere

__all__ = ['mcdnvcycle', 'mcdnsolver']

import numpy as np
from monitor import indentprint, ObstacleMonitor

def _levelreport(indent, j, m, sweeps):
    indentprint(indent - j, 'level %d: %d sweeps over m=%d nodes' \
                             % (j, sweeps, m))

def _coarsereport(indent, m, sweeps):
    indentprint(indent, 'coarsest: %d sweeps over m=%d nodes' \
                         % (sweeps, m))

def mcdnvcycle(args, obsprob, J, hierarchy, w, ell, levels=None):
    '''Apply one V-cycle of the nonlinear MCD method.  Input args is a
    dictionary with parameters.  Input obsprob is of type
    SmootherObstacleProblem.  The smoother is projected nonlinear Gauss-Seidel
    or projected nonlinear Jacobi according to args.jacobi.  Note hierarchy[j]
    is of type MeshLevel1D.  This method generates all the defect constraints hierarchy[j].chi for j < J, but it requires hierarchy[J].chi to be set in
    advance.  Note that hierarchy[j].b must also be defined for
    all mesh levels.  The input iterate w is in V^j and the input linear
    functional ell is in V^J'.  The coarse solver is the same as the smoother.'''

    # set up
    assert args.down >= 0 and args.up >= 0 and args.coarse >= 0
    hierarchy[J].checklen(w)
    hierarchy[J].checklen(ell)
    hierarchy[J].ell = ell
    hierarchy[J].g = w

    # downward
    for k in range(J, 0, -1):
        # compute new bed and defect constraint using monotone restriction
        hierarchy[k-1].chi = hierarchy[k].mR(hierarchy[k].chi)
        # define down-obstacle
        phi = hierarchy[k].chi - hierarchy[k].cP(hierarchy[k-1].chi)
        # down-smooth the correction y
        if args.mgview:
            _levelreport(levels-1, k, hierarchy[k].m, args.down)
        hierarchy[k].y = hierarchy[k].zeros()
        obsprob.smoother(args.down, hierarchy[k], hierarchy[k].y,
                         hierarchy[k].ell, phi, symmetric=args.symmetric)
        # determine fixed part of solution on next level down
        wk = hierarchy[k].g + hierarchy[k].y
        hierarchy[k-1].g = hierarchy[k].iR(wk)
        # update residual and determine source on next level down
        F = obsprob.residual(hierarchy[k], wk, hierarchy[k].ell)
        hierarchy[k-1].ell = obsprob.applyN(hierarchy[k-1], hierarchy[k-1].g) \
                             - hierarchy[k].cR(F)

    # coarse mesh solver = smoother sweeps on correction y
    if args.mgview:
        _coarsereport(levels-1, hierarchy[0].m, args.coarse)
    hierarchy[0].y = hierarchy[0].zeros()
    obsprob.smoother(args.coarse, hierarchy[0], hierarchy[0].y,
                     hierarchy[0].ell, hierarchy[0].chi, symmetric=args.symmetric)

    # upward
    z = hierarchy[0].y
    for k in range(1, J+1):
        # accumulate corrections
        z = hierarchy[k].cP(z) + hierarchy[k].y
        if args.up > 0:
            # up-smooth corrections y;  obstacle is chi[k] not phi (see paper)
            if args.mgview:
                _levelreport(levels-1, k, hierarchy[k].m, args.up)
            obsprob.smoother(args.up, hierarchy[k], z,
                             hierarchy[k].ell, hierarchy[k].chi,
                             symmetric=args.symmetric, forward=False)
    return z

def mcdnsolver(args, obsprob, J, hierarchy, ella, b, w, monitor,
               iters=100, irnorm0=None):
    '''Apply V-cycles of the MCDN method until convergence by an inactive
    residual norm tolerance.  Calls mcdnvcycle().'''

    mesh = hierarchy[J]
    mesh.checklen(ella)
    mesh.checklen(b)
    mesh.checklen(w)
    if irnorm0 == None:
        irnorm0, _ = monitor.irerr(w, ella, b, indent=0)
    if irnorm0 == 0.0:
        return
    for s in range(iters):
        mesh.b = b
        mesh.chi = b - w                       # defect obstacle
        w += mcdnvcycle(args, obsprob, J, hierarchy, w, ella, levels=J+1)  # FIXME: not obviously the correct args
        irnorm, errnorm = monitor.irerr(w, ella, b, indent=0)
        if irnorm > 100.0 * irnorm0:
            print('WARNING:  irnorm > 100 irnorm0')
        if irnorm <= args.irtol * irnorm0:
            break
