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

#FIXME keep recursive?
def mcdlslash(j, hierarchy, ell, down=1, coarse=1,
              levels=None, view=False, symmetric=False, printwarnings=False):
    '''Apply one cycle of the multilevel subset decomposition method of
    Tai (2003), as stated in Alg. 4.7 in Graeser & Kornhuber (2009),
    but implemented recursively.  Note hierarchy[j] is of type MeshLevel1D,
    and hierarch[j].chi is the jth-level defect constraint.  Linear functional
    ell is in V^J'.  The smoother is projected Gauss-Seidel (PGS).  The coarse
    solver is coarse iterations of PGS, thus not exact.'''

    # set up
    mesh = hierarchy[j]
    y = mesh.zeros()
    assert down >= 1 and coarse >= 1 and len(ell) == mesh.m + 2

    # coarse mesh solver = PGS sweeps
    if j == 0:
        if view:
            _coarsereport(levels-1, mesh.m, coarse)
        infeas = _smoother(coarse, mesh, y, ell, mesh.chi,
                           symmetric=symmetric, printwarnings=printwarnings)
        return y, infeas

    # update defect constraint and define jth-level obstacle
    hierarchy[j-1].chi = mesh.mR(mesh.chi)
    phi = mesh.chi - mesh.P(hierarchy[j-1].chi)

    # down smoother = PGS sweeps
    if view:
        _levelreport(levels-1, j, mesh.m, down)
    infeas = _smoother(down, mesh, y, ell, phi,
                       symmetric=symmetric, printwarnings=printwarnings)

    # canonically-restrict the residual and update to F^{j-1}(y)[.]
    ellcoarse = - mesh.cR(residual(mesh, y, ell))

    # recursive coarse-level correction
    ycoarse, ifc = mcdlslash(j-1, hierarchy, ellcoarse,
                             down=down, coarse=coarse,
                             levels=levels, view=view, symmetric=symmetric,
                             printwarnings=printwarnings)
    y += mesh.P(ycoarse)
    infeas += ifc
    return y, infeas

