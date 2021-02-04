'''Module implementing the V-cycle algorithm for the Tai (2003) multilevel
subset decomposition method.'''

from poisson import residual
from pgs import pgssweep

__all__ = ['vcycle']

def _indentprint(n, s):
    '''Print 2n spaces and then string s.'''
    for _ in range(n):
        print('  ', end='')
    print(s)

def _levelreport(indent, k, m, sweeps):
    _indentprint(indent - k, 'level %d: %d sweeps over m=%d nodes' \
                              % (k, sweeps, m))

def _coarsereport(indent, m, sweeps):
    _indentprint(indent, 'coarsest: %d sweeps over m=%d nodes' \
                         % (sweeps, m))

def _smoother(s, mesh, v, r, phi, forward=True, symmetric=False, printwarnings=False):
    infeas = 0
    for _ in range(s):
        infeas += pgssweep(mesh, v, r, phi, forward=forward,
                           printwarnings=printwarnings)
        if symmetric:
            infeas += pgssweep(mesh, v, r, phi, forward=not forward,
                               printwarnings=printwarnings)
    return infeas

def vcycle(k, hierarchy, r, down=1, up=0, coarse=1,
           levels=None, view=False, symmetric=False, printwarnings=False):
    '''Apply one V-cycle of the multilevel subset decomposition method of
    Tai (2003).  This is  Alg. 4.7 in Graeser & Kornhuber (2009) when up=0.
    This solves the defect constraint problem on mesh = hierarchy[k], i.e.
    for chi^k = hierarchy[k].chi.  Note hierarchy[k] is of type MeshLevel1D.
    Right-hand-side r is in the fine-mesh linear functional space (V^K)'.
    The smoother is down, up iterations of projected Gauss-Seidel (PGS).
    The coarse solver is coarse iterations of PGS (thus not exact).'''

    # set up
    assert down >= 1 and up >= 0 and coarse >= 1
    mesh = hierarchy[k]
    assert len(r) == mesh.m + 2
    v = mesh.zeros()

    # coarse mesh solver = PGS sweeps
    if k == 0:
        if view:
            _coarsereport(levels-1, mesh.m, coarse)
        infeas = _smoother(coarse, mesh, v, r, mesh.chi,
                           symmetric=symmetric, printwarnings=printwarnings)
        return v, infeas

    # monotone restriction of defect constraint
    hierarchy[k-1].chi = mesh.mR(mesh.chi)
    # level k obstacle is the *change* in chi
    phi = mesh.chi - mesh.P(hierarchy[k-1].chi)
    if up > 0:
        phi *= 0.5
    # down smoother = PGS sweeps
    if view:
        _levelreport(levels-1, k, mesh.m, down)
    infeas = _smoother(down, mesh, v, r, phi,
                       symmetric=symmetric, printwarnings=printwarnings)
    # update and canonically-restrict the residual
    rcoarse = mesh.cR(residual(mesh, v, r))
    # coarse-level correction
    vcoarse, ifc = vcycle(k-1, hierarchy, rcoarse,
                          down=down, up=up, coarse=coarse,
                          levels=levels, view=view, symmetric=symmetric,
                          printwarnings=printwarnings)
    v += mesh.P(vcoarse)
    infeas += ifc
    # up smoother = PGS sweeps
    if up > 0:
        if view:
            _levelreport(levels-1, k, mesh.m, up)
        infeas += _smoother(up, mesh, v, r, phi,
                            symmetric=symmetric, printwarnings=printwarnings)
    return v, infeas
