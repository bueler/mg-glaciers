'''Module implementing the multilevel constraint decomposition (MCD) method
of the Tai (2003).'''

from poisson import residual
from pgs import pgssweep

__all__ = ['mcdlslash']

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

def mcdlslash(J, hierarchy, ell, down=1, coarse=1,
              levels=None, view=False, symmetric=False, printwarnings=False):
    '''Apply one cycle of the multilevel subset decomposition method of
    Tai (2003), as stated in Alg. 4.7 in Graeser & Kornhuber (2009).
    Note hierarchy[j] is of type MeshLevel1D, and hierarchy[j].chi is the
    jth-level defect constraint.  Input linear functional ell is in V^J'.
    The smoother is projected Gauss-Seidel (PGS).  The coarse solver is PGS,
    thus not exact.'''

    # set up
    assert down >= 1 and coarse >= 1 and len(ell) == hierarchy[J].m + 2
    infeas = 0

    # downward
    for k in range(J,0,-1):
        # update defect constraint and define obstacle
        hierarchy[k-1].chi = hierarchy[k].mR(hierarchy[k].chi)
        phi = hierarchy[k].chi - hierarchy[k].P(hierarchy[k-1].chi)
        # down smoother = PGS sweeps
        if view:
            _levelreport(levels-1, k, hierarchy[k].m, down)
        hierarchy[k].y = hierarchy[k].zeros()
        infeas += _smoother(down, hierarchy[k], hierarchy[k].y, ell, phi,
                            symmetric=symmetric, printwarnings=printwarnings)
        # canonically-restrict the residual
        ell = - hierarchy[k].cR(residual(hierarchy[k], hierarchy[k].y, ell))

    # coarse mesh solver = PGS sweeps
    if view:
        _coarsereport(levels-1, hierarchy[0].m, coarse)
    hierarchy[0].y = hierarchy[0].zeros()
    infeas += _smoother(coarse, hierarchy[0], hierarchy[0].y, ell,
                        hierarchy[0].chi,
                        symmetric=symmetric, printwarnings=printwarnings)

    # upward
    for k in range(1,J+1):
        hierarchy[k].y += hierarchy[k].P(hierarchy[k-1].y)
    return hierarchy[J].y, infeas

