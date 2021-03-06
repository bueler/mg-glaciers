'''Module implementing the multilevel constraint decomposition (MCD) method
for the classical obstacle problem, i.e. for a linear interior PDE like the
Poisson equation.'''

__all__ = ['mcdlvcycle', 'mcdlfcycle', 'mcdlsolver']

import numpy as np
from monitor import indentprint, ObstacleMonitor

def _levelreport(indent, j, m, sweeps):
    indentprint(indent - j, 'level %d: %d sweeps over m=%d nodes' \
                             % (j, sweeps, m))

def _coarsereport(indent, m, sweeps):
    indentprint(indent, 'coarsest: %d sweeps over m=%d nodes' \
                         % (sweeps, m))

def mcdlvcycle(args, obsprob, J, hierarchy, ell, levels=None):
    '''Apply one V-cycle of the multilevel constraint decomposition method of
    Tai (2003).  This is stated in Alg. 4.7 in Graeser & Kornhuber (2009)
    as a down-slash V(1,0) cycle.  Our implementation allows any V(down,up)
    cycle.  Input args is a dictionary with parameters.  Input obsprob is
    of type SmootherObstacleProblem.  The smoother is projected Gauss-Seidel
    or projected Jacobi according to args.jacobi.
    Note hierarchy[j] is of type MeshLevel1D.  This method generates all defect
    constraints hierarchy[j].chi for j < J, but it uses hierarchy[J].chi, which
    must be set in advance.  The input linear functional ell is in V^J'.
    The coarse solver is the same as the smoother, thus not exact.'''

    # set up
    assert args.down >= 0 and args.up >= 0 and args.coarse >= 0
    hierarchy[J].checklen(ell)
    hierarchy[J].ell = ell

    # downward
    for k in range(J, 0, -1):
        # compute defect constraint using monotone restriction
        hierarchy[k-1].chi = hierarchy[k].mR(hierarchy[k].chi)
        # define down-obstacle
        phi = hierarchy[k].chi - hierarchy[k].cP(hierarchy[k-1].chi)
        # down smoother
        if args.mgview:
            _levelreport(levels-1, k, hierarchy[k].m, args.down)
        hierarchy[k].y = hierarchy[k].zeros()
        obsprob.smoother(args.down, hierarchy[k], hierarchy[k].y,
                         hierarchy[k].ell, phi)
        # update and canonically-restrict the residual
        hierarchy[k-1].ell = - hierarchy[k].cR(obsprob.residual(hierarchy[k],
                                                                hierarchy[k].y,
                                                                hierarchy[k].ell))

    # coarse mesh solver = smoother sweeps
    if args.mgview:
        _coarsereport(levels-1, hierarchy[0].m, args.coarse)
    hierarchy[0].y = hierarchy[0].zeros()
    obsprob.smoother(args.coarse, hierarchy[0], hierarchy[0].y,
                     hierarchy[0].ell, hierarchy[0].chi)

    # upward
    z = hierarchy[0].y
    for k in range(1, J+1):
        # accumulate corrections
        z = hierarchy[k].cP(z) + hierarchy[k].y
        if args.up > 0:
            # up smoother; up-obstacle is chi[k] not phi (see paper)
            if args.mgview:
                _levelreport(levels-1, k, hierarchy[k].m, args.up)
            obsprob.smoother(args.up, hierarchy[k], z,
                             hierarchy[k].ell, hierarchy[k].chi)
    return z

def mcdlfcycle(args, obsprob, J, hierarchy):
    '''Apply an F-cycle, i.e. nested iteration, of the multilevel constraint
    decomposition method of Tai (2003).  This method calls mcdlvcycle().'''

    assert args.ni
    phi = obsprob.phi(hierarchy[0].xx())
    w = phi.copy()
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
        ellf = mesh.ellf(obsprob.source(mesh.xx()))  # source functional ell[v] = <f,v>
        for s in range(iters):
            mon.irerr(w, ellf, phi, indent=J-j)       # print norms at stdout
            mesh.chi = phi - w                        # defect obstacle
            ell = - obsprob.residual(mesh, w, ellf)   # starting source
            w += mcdlvcycle(args, obsprob, j, hierarchy, ell, levels=j+1)
        mon.irerr(w, ellf, phi, indent=J-j)
        # obstacle and initial iterate for next level; prolong and truncate current solution
        if j < J:
            phi = obsprob.phi(hierarchy[j+1].xx())
            w = np.maximum(phi, hierarchy[j+1].cP(w))
    return w

def mcdlsolver(args, obsprob, J, hierarchy, ellf, phi, w, monitor,
               iters=100, irnorm0=None):
    '''Apply V-cycles of the multilevel constraint decomposition method of
    Tai (2003) until convergence by an inactive residual norm tolerance.
    This method calls mcdlvcycle().'''

    mesh = hierarchy[J]
    if irnorm0 == None:
        irnorm0, _ = monitor.irerr(w, ellf, phi, indent=0)
    if irnorm0 == 0.0:
        return
    for s in range(iters):
        mesh.chi = phi - w                       # defect obstacle
        ell = - obsprob.residual(mesh, w, ellf)  # starting source
        w += mcdlvcycle(args, obsprob, J, hierarchy, ell, levels=J+1)
        irnorm, errnorm = monitor.irerr(w, ellf, phi, indent=0)
        if irnorm > 100.0 * irnorm0:
            print('WARNING:  irnorm > 100 irnorm0')
        if irnorm <= args.irtol * irnorm0:
            break
