'''Module implementing the multilevel constraint decomposition (MCD) method
of the Tai (2003) for the classical obstacle problem (i.e. linear interior PDE).'''

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

def _smoother(obsprob, s, mesh, v, ell, phi, forward=True, symmetric=False):
    for _ in range(s):
        obsprob.smoothersweep(mesh, v, ell, phi, forward=forward)
        if symmetric:
            obsprob.smoothersweep(mesh, v, ell, phi, forward=not forward)

def mcdlcycle(obsprob, J, hierarchy, ell, down=1, up=0, coarse=1,
              levels=None, view=False, symmetric=False):
    '''Apply one cycle of the multilevel subset decomposition method of
    Tai (2003).  This os stated in Alg. 4.7 in Graeser & Kornhuber (2009)
    as a down-slash cycle (down=1, up=0).  Our implementation allows any
    V(down,up) cycle.  Note hierarchy[j] is of type MeshLevel1D, and
    this method generates all defect constraints hierarchy[j].chi.  The input
    linear functional ell is in V^J'.  The smoother is projected Gauss-Seidel
    or projected Jacobi.  The coarse solver is the same as the smoother,
    thus not exact.'''

    # set up
    assert down >= 0 and up >= 0 and coarse >= 0
    assert len(ell) == hierarchy[J].m + 2
    hierarchy[J].ell = ell

    # downward
    for k in range(J, 0, -1):
        # compute defect constraint using monotone restriction
        hierarchy[k-1].chi = hierarchy[k].mR(hierarchy[k].chi)
        # define down-obstacle
        phi = hierarchy[k].chi - hierarchy[k].cP(hierarchy[k-1].chi)
        # down smoother
        if view:
            _levelreport(levels-1, k, hierarchy[k].m, down)
        hierarchy[k].y = hierarchy[k].zeros()
        _smoother(obsprob, down, hierarchy[k], hierarchy[k].y,
                  hierarchy[k].ell, phi, symmetric=symmetric)
        # update and canonically-restrict the residual
        hierarchy[k-1].ell = - hierarchy[k].cR(obsprob.residual(hierarchy[k],
                                                                hierarchy[k].y,
                                                                hierarchy[k].ell))

    # coarse mesh solver = smoother sweeps
    if view:
        _coarsereport(levels-1, hierarchy[0].m, coarse)
    hierarchy[0].y = hierarchy[0].zeros()
    _smoother(obsprob, coarse, hierarchy[0], hierarchy[0].y,
              hierarchy[0].ell, hierarchy[0].chi, symmetric=symmetric)

    # upward
    z = hierarchy[0].y
    for k in range(1, J+1):
        # accumulate corrections
        z = hierarchy[k].cP(z) + hierarchy[k].y
        if up > 0:
            # up smoother; up-obstacle is chi[k] not phi (see paper)
            if view:
                _levelreport(levels-1, k, hierarchy[k].m, up)
            _smoother(obsprob, up, hierarchy[k], z,
                      hierarchy[k].ell, hierarchy[k].chi,
                      symmetric=symmetric, forward=False)

    return z
