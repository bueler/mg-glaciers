'''Module implementing the multilevel constraint decomposition (MCD) method
of the Tai (2003).'''

from poisson import residual
from pgs import pgssweep

__all__ = ['mcdlcycle']

def _indentprint(n, s):
    '''Print 2n spaces and then string s.'''
    for _ in range(n):
        print('  ', end='')
    print(s)

def _levelreport(indent, j, m, sweeps):
    _indentprint(indent - j, 'level %d: %d sweeps over m=%d nodes' \
                             % (j, sweeps, m))

def _coarsereport(indent, m, sweeps):
    _indentprint(indent, 'coarsest: %d sweeps over m=%d nodes' \
                         % (sweeps, m))

def _smoother(s, mesh, v, ell, phi, forward=True, symmetric=False, printwarnings=False):
    infeas = 0
    for _ in range(s):
        infeas += pgssweep(mesh, v, ell, phi, forward=forward,
                           printwarnings=printwarnings)
        if symmetric:
            infeas += pgssweep(mesh, v, ell, phi, forward=not forward,
                               printwarnings=printwarnings)
    return infeas

def mcdlcycle(J, hierarchy, ell, down=1, up=1, coarse=1,
              levels=None, view=False, symmetric=False, printwarnings=False):
    '''Apply one cycle of the multilevel subset decomposition method of
    Tai (2003), as stated in Alg. 4.7 in Graeser & Kornhuber (2009),
    either as a slash cycle (up=0) or as a V-cycle (up>0).
    Note hierarchy[j] is of type MeshLevel1D, and hierarchy[j].chi is the
    jth-level defect constraint.  Input linear functional ell is in V^J'.
    The smoother is projected Gauss-Seidel (PGS).  The coarse solver is PGS,
    thus not exact.'''

    # set up
    assert down >= 0 and up >=0 and coarse >= 0
    assert len(ell) == hierarchy[J].m + 2
    infeas = 0
    hierarchy[J].ell = ell

    # downward
    for k in range(J,0,-1):
        # update defect constraint and define obstacle
        hierarchy[k-1].chi = hierarchy[k].mR(hierarchy[k].chi)
        phi = hierarchy[k].chi - hierarchy[k].cP(hierarchy[k-1].chi)
        if up > 0:
            phi *= 0.5  # V-cycle obstacle
        # down smoother = PGS sweeps
        if view:
            _levelreport(levels-1, k, hierarchy[k].m, down)
        hierarchy[k].y = hierarchy[k].zeros()
        infeas += _smoother(down, hierarchy[k], hierarchy[k].y, hierarchy[k].ell, phi,
                            symmetric=symmetric, printwarnings=printwarnings)
        # update and canonically-restrict the residual
        hierarchy[k-1].ell = - hierarchy[k].cR(residual(hierarchy[k],
                                                        hierarchy[k].y,
                                                        hierarchy[k].ell))

    # coarse mesh solver = PGS sweeps
    if view:
        _coarsereport(levels-1, hierarchy[0].m, coarse)
    hierarchy[0].y = hierarchy[0].zeros()
    infeas += _smoother(coarse, hierarchy[0], hierarchy[0].y, hierarchy[0].ell,
                        hierarchy[0].chi,
                        symmetric=symmetric, printwarnings=printwarnings)

    # upward
    hierarchy[0].omega = hierarchy[0].y.copy()
    for k in range(1,J+1):
        # accumulate corrections
        hierarchy[k].omega = hierarchy[k].cP(hierarchy[k-1].omega) + hierarchy[k].y
        if up > 0:
            # V-cycle obstacle
            phi = 0.5 * (hierarchy[k].chi - hierarchy[k].cP(hierarchy[k-1].chi))
            # update and prolong the residual
            # FIXME at least 2 possibilities, but neither work properly
            #hierarchy[k].ell = - hierarchy[k].injectP(residual(hierarchy[k-1],
            #                                                   hierarchy[k-1].y,
            #                                                   hierarchy[k-1].ell))
            hierarchy[k].ell = - residual(hierarchy[k],
                                          hierarchy[k].cP(hierarchy[k-1].y),
                                          hierarchy[k].injectP(hierarchy[k-1].ell))
            # up smoother = PGS sweeps
            if view:
                _levelreport(levels-1, k, hierarchy[k].m, up)
            hierarchy[k].y = hierarchy[k].zeros()
            infeas += _smoother(up, hierarchy[k], hierarchy[k].y, hierarchy[k].ell, phi,
                                symmetric=symmetric, printwarnings=printwarnings,
                                forward=False)
            hierarchy[k].omega += hierarchy[k].y

    return hierarchy[J].omega, infeas

