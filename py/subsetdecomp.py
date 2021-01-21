# module to implement the V-cycle algorithm for the Tai (2003)
#     multilevel subset decomposition method

from poisson import residual
from pgs import pgssweep
import numpy as np

__all__ = ['vcycle']

def _indentprint(n,s):
    '''Print 2n spaces and then string s.'''
    for i in range(n):
        print('  ',end='')
    print(s)

def _levelreport(indent,k,m,sweeps):
    _indentprint(indent-k,'level %d: %d sweeps over m=%d nodes' \
                          % (k,sweeps,m))

def _coarsereport(indent,m,sweeps):
    _indentprint(indent,'coarsest: %d sweeps over m=%d nodes' \
                        % (sweeps,m))

def vcycle(hierarchy,w,ell,phi,
           levels=None,view=False,symmetric=False,
           down=1,coarse=1,up=0):
    '''Apply one V(1,0)-cycle of the multilevel subset decomposition
    method from Tai (2003), namely Alg. 4.7 in Graeser & Kornhuber (2009).
    In-place updates the iterate w to solve the obstacle problem on the
    mesh = hierarchy[-1] = hierarchy[levels-1].  Vectors w,phi are in the
    fine-mesh function space (V^K) while ell is a fine-mesh linear functional
    (in (V^K)').  Note hierarchy[0],...,hierarchy[levels-1]
    (coarse to fine) are of type MeshLevel1D.  The smoother is down
    iterations of projected Gauss-Seidel (pGS).  The coarse solver is
    coarse iterations of pGS (thus possibly not exact).'''

    if up > 0:
        raise NotImplementedError

    mesh = hierarchy[-1]
    assert (len(w) == mesh.m+2)
    assert (len(ell) == mesh.m+2)
    assert (len(phi) == mesh.m+2)
    chi = [None] * (levels)           # empty list
    fine = levels - 1

    # the only place w is used:
    chi[fine] = phi - w               # fine mesh defect obstacle
    r = residual(mesh,w,ell)          # fine mesh residual

    # DOWN
    for k in range(fine,0,-1):        # k=fine,fine-1,...,1
        # monotone restriction decomposes defect obstacle
        chi[k-1] = hierarchy[k].mR(chi[k])
        # the level k obstacle is the *change* in chi
        Psi = chi[k] - hierarchy[k].P(chi[k-1])
        # do projected GS sweeps
        if view:
            _levelreport(fine,k,hierarchy[k].m,down)
        v = hierarchy[k].zeros()
        for s in range(down):
            pgssweep(hierarchy[k],v,r,Psi)
            if symmetric:
                pgssweep(hierarchy[k],v,r,Psi,forward=False)
        hierarchy[k].vstate = v.copy()
        # update and canonically-restrict the residual
        r = residual(hierarchy[k],v,r)
        r = hierarchy[k].cR(r)
    # COARSE SOLVE
    Psi = chi[0]
    if view:
        _coarsereport(fine,hierarchy[0].m,coarse)
    v = hierarchy[0].zeros()
    for s in range(coarse):
        pgssweep(hierarchy[0],v,r,Psi)
        if symmetric:
            pgssweep(hierarchy[0],v,r,Psi,forward=False)
    hierarchy[0].vstate = v.copy()
    # UP
    for k in range(1,fine+1):        # k=1,2,...,fine
        if view:
            _levelreport(fine,k,hierarchy[k].m,up)
        hierarchy[k].vstate += hierarchy[k].P(hierarchy[k-1].vstate)
    # new iterate
    w += hierarchy[fine].vstate
    return w, chi

