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

def vcycle(hierarchy, w, ell, phi,
           levels=None, view=False, symmetric=False,
           down=1, coarse=1, up=0, printwarnings=False):
    '''Apply one V(1,0)-cycle of the multilevel subset decomposition
    method from Tai (2003), namely Alg. 4.7 in Graeser & Kornhuber (2009).
    In-place updates the iterate w to solve the obstacle problem on the
    mesh = hierarchy[-1] = hierarchy[levels-1].  Vectors w,phi are in the
    fine-mesh function space (V^K) while ell is a fine-mesh linear functional
    (in (V^K)').  Note hierarchy[0],...,hierarchy[levels-1]
    (coarse to fine) are of type MeshLevel1D.  The smoother is down
    iterations of projected Gauss-Seidel (pGS).  The coarse solver is
    coarse iterations of pGS (thus possibly not exact).'''

    assert down >= 0 and coarse >= 0 and up >= 0

    mesh = hierarchy[-1]
    assert len(w) == mesh.m + 2
    assert len(ell) == mesh.m + 2
    assert len(phi) == mesh.m+2
    infeas = 0

    # the only place w is used:
    mesh.chi = phi - w                        # fine mesh defect constraint
    hierarchy[-1].r = residual(mesh, w, ell)  # fine mesh residual

    # DOWN
    for k in range(levels-1, 0, -1):          # k=levels-1,levels-2,...,1
        if view:
            _levelreport(levels-1, k, hierarchy[k].m, down)
        # monotone restriction decomposes defect obstacle
        hierarchy[k-1].chi = hierarchy[k].mR(hierarchy[k].chi)
        # the level k obstacle is the *change* in chi
        Psi = hierarchy[k].chi - hierarchy[k].P(hierarchy[k-1].chi)
        if up == 1:
            Psi *= 0.5
        # do projected GS sweeps
        v = hierarchy[k].zeros()
        for _ in range(down):
            infeas += pgssweep(hierarchy[k], v, hierarchy[k].r, Psi,
                               printwarnings=printwarnings)
            if symmetric:
                infeas += pgssweep(hierarchy[k], v, hierarchy[k].r, Psi,
                                   forward=False, printwarnings=printwarnings)
        hierarchy[k].vstate = v.copy()
        # update and canonically-restrict the residual
        hierarchy[k].r = residual(hierarchy[k], v, hierarchy[k].r)
        hierarchy[k-1].r = hierarchy[k].cR(hierarchy[k].r)

    # COARSE SOLVE
    if view:
        _coarsereport(levels-1, hierarchy[0].m, coarse)
    Psi = hierarchy[0].chi
    v = hierarchy[0].zeros()
    for _ in range(coarse):
        infeas += pgssweep(hierarchy[0], v, hierarchy[0].r, Psi,
                           printwarnings=printwarnings)
        if symmetric:
            infeas += pgssweep(hierarchy[0], v, hierarchy[0].r, Psi,
                               printwarnings=printwarnings, forward=False)
    hierarchy[0].vstate = v.copy()

    # UP
    for k in range(1, levels):        # k=1,2,...,levels-1
        if view:
            _levelreport(levels-1, k, hierarchy[k].m, up)
        if up >= 1:
            # the current iterate is what came back from k-1 level (WHY?)
            v = hierarchy[k].P(hierarchy[k-1].vstate)
            # the level k obstacle is the *change* in chi
            Psi = 0.5 * (hierarchy[k].chi - hierarchy[k].P(hierarchy[k-1].chi))
            # do projected GS sweeps
            for _ in range(up):
                infeas += pgssweep(hierarchy[k], v, hierarchy[k].r, Psi,
                                   printwarnings=printwarnings, forward=False)
                if symmetric:
                    infeas += pgssweep(hierarchy[k], v, hierarchy[k].r, Psi,
                                       printwarnings=printwarnings)
            hierarchy[k].vstate += v
        else:
            hierarchy[k].vstate += hierarchy[k].P(hierarchy[k - 1].vstate)

    # new iterate
    w += hierarchy[-1].vstate
    return w, infeas

def slash(k, hierarchy, r, down=1, coarse=1,
          levels=None, view=False, symmetric=False, printwarnings=False):
    '''Apply one V(1,0)-cycle of the multilevel subset decomposition
    method from Tai (2003), namely Alg. 4.7 in Graeser & Kornhuber (2009).
    This solves the defect constraint problem on mesh = hierarchy[k], i.e.
    for chi^k = hierarchy[k].chi.  Note hierarchy[k] is of type MeshLevel1D.
    Right-hand-side r is in the fine-mesh linear functional space (V^K)'.
    The smoother is down iterations of projected Gauss-Seidel (PGS).  The
    coarse solver is coarse iterations of PGS (thus possibly not exact).'''

    # set up
    assert down >= 0 and coarse >= 0
    mesh = hierarchy[k]
    assert len(r) == mesh.m + 2
    infeas = 0
    v = mesh.zeros()
    if view:
        if k == 0:
            _coarsereport(levels-1, mesh.m, coarse)
        else:
            _levelreport(levels-1, k, mesh.m, down)

    # coarse mesh solver = PGS sweeps
    if k == 0:
        for _ in range(coarse):
            infeas += pgssweep(mesh, v, r, mesh.chi,
                               printwarnings=printwarnings)
            if symmetric:
                infeas += pgssweep(mesh, v, r, mesh.chi, forward=False,
                                   printwarnings=printwarnings)
        return v, infeas

    # monotone restriction of defect constraint
    hierarchy[k-1].chi = mesh.mR(mesh.chi)
    # level k obstacle is the *change* in chi
    phi = mesh.chi - mesh.P(hierarchy[k-1].chi)
    # smoother = PGS sweeps
    for _ in range(down):
        infeas += pgssweep(mesh, v, r, phi,
                           printwarnings=printwarnings)
        if symmetric:
            infeas += pgssweep(mesh, v, r, phi, forward=False,
                               printwarnings=printwarnings)
    # update and canonically-restrict the residual
    rcoarse = mesh.cR(residual(mesh, v, r))
    # coarse-level correction
    vcoarse, ifc = slash(k-1, hierarchy, rcoarse, down=down, coarse=coarse,
                         levels=levels, view=view, symmetric=symmetric,
                         printwarnings=printwarnings)
    return v + mesh.P(vcoarse), infeas + ifc
