'''Module implementing the nonlinear full approximation storage (FAS) extension
of the multilevel constraint decomposition (MCD) method for the shallow ice
approximation (SIA) obstacle problem.'''

# TODO:
#   * compare write-up to functions below
#   * implement p=4 p-Laplacian obstacle problem; how much better?
#   * convergence and performance studies
#   * implement random-bed case (for performance only)
#   * implement multiple ice sheets case (for performance only; requires e.g. -jcoarse 4)

# PERFORMANCE QUESTIONS:
#   * does +eps in thickness coefficient in N(w)[v] make things better?
#     (PRELIMINARY ANSWER: no)
#   * does monotone increase (or decrease) on thickness coefficient in N(w)[v]
#     on coarsening make things better?
#   * how valuable are really-accurate solves on the coarsest levels?
#     (PRELIMINARY ANSWER: within V-cycle, not important, but e.g. -ni -nicycles 16 is a good idea)

__all__ = ['mcdnvcycle', 'mcdnfcycle', 'mcdnsolver']

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

def mcdnfcycle(args, obsprob, J, hierarchy):
    '''Apply an MCDN F-cycle, i.e. nested iteration.  This method calls
    mcdnvcycle().  Compare mcdlfcycle().'''

    assert args.ni
    phi = obsprob.phi(hierarchy[0].xx())
    w = obsprob.initial(hierarchy[0].xx())
    for j in range(J+1):
        mesh = hierarchy[j]
        # create monitor on this mesh using exact solution if available
        uex = None
        if obsprob.exact_available():
            uex = obsprob.exact(mesh.xx())
        mon = ObstacleMonitor(obsprob, mesh, uex=uex,
                              printresiduals=args.monitor, printerrors=args.monitorerr)
        # how many cycles?
        iters = args.nicycles
        if args.nicascadic:
            # very simple model for number of cycles; compare Blum et al 2004
            iters *= int(np.ceil(1.5**(J-j)))
        # do V cycles
        mesh.b = obsprob.phi(mesh.xx())
        ella = mesh.ellf(obsprob.source(mesh.xx()))  # source functional ell[v] = <f,v>
        for s in range(iters):
            mon.irerr(w, ella, phi, indent=J-j)       # print norms at stdout
            mesh.chi = phi - w                        # defect obstacle
            w += mcdnvcycle(args, obsprob, j, hierarchy, w, ella, levels=j+1)
        mon.irerr(w, ella, phi, indent=J-j)
        # obstacle and initial iterate for next level; prolong and truncate current solution
        if j < J:
            phi = obsprob.phi(hierarchy[j+1].xx())
            w = np.maximum(phi, hierarchy[j+1].cP(w))
    return w

def mcdnsolver(args, obsprob, J, hierarchy, ella, b, w, monitor,
               iters=100, irnorm0=None):
    '''Apply V-cycles of the MCDN method until convergence by an inactive
    residual norm tolerance.  Calls mcdnvcycle().  Compare mcdlsolver().'''

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
        mesh.chi = b - w
        w += mcdnvcycle(args, obsprob, J, hierarchy, w, ella, levels=J+1)
        irnorm, errnorm = monitor.irerr(w, ella, b, indent=0)
        if irnorm > 100.0 * irnorm0:
            print('WARNING:  irnorm > 100 irnorm0')
        if irnorm <= args.irtol * irnorm0:
            break
