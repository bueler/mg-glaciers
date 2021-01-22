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

    assert down >= 0 and coarse >= 0 and up >= 0

    mesh = hierarchy[-1]
    assert (len(w) == mesh.m+2)
    assert (len(ell) == mesh.m+2)
    assert (len(phi) == mesh.m+2)
    chi = [None] * (levels)           # empty list

    # the only place w is used:
    chi[-1] = phi - w                       # fine mesh defect obstacle
    hierarchy[-1].r = residual(mesh,w,ell)  # fine mesh residual

    # DOWN
    for k in range(levels-1,0,-1):          # k=levels-1,levels-2,...,1
        if view:
            _levelreport(levels-1,k,hierarchy[k].m,down)
        # monotone restriction decomposes defect obstacle
        chi[k-1] = hierarchy[k].mR(chi[k])
        # the level k obstacle is the *change* in chi
        Psi = chi[k] - hierarchy[k].P(chi[k-1])
        if up == 1:
            Psi *= 0.5
        # do projected GS sweeps
        v = hierarchy[k].zeros()
        for s in range(down):
            pgssweep(hierarchy[k],v,hierarchy[k].r,Psi)
            if symmetric:
                pgssweep(hierarchy[k],v,hierarchy[k].r,Psi,forward=False)
        hierarchy[k].vstate = v.copy()
        # update and canonically-restrict the residual
        hierarchy[k].r = residual(hierarchy[k],v,hierarchy[k].r)
        hierarchy[k-1].r = hierarchy[k].cR(hierarchy[k].r)

    # COARSE SOLVE
    if view:
        _coarsereport(levels-1,hierarchy[0].m,coarse)
    Psi = chi[0]
    v = hierarchy[0].zeros()
    for s in range(coarse):
        pgssweep(hierarchy[0],v,hierarchy[0].r,Psi)
        if symmetric:
            pgssweep(hierarchy[0],v,hierarchy[0].r,Psi,forward=False)
    hierarchy[0].vstate = v.copy()

    # UP
    for k in range(1,levels):        # k=1,2,...,levels-1
        if view:
            _levelreport(levels-1,k,hierarchy[k].m,up)
        if up == 1:
            # the current iterate is what came back from k-1 level (WHY?)
            v = hierarchy[k].P(hierarchy[k-1].vstate)
            # the level k obstacle is the *change* in chi
            Psi = 0.5 * (chi[k] - hierarchy[k].P(chi[k-1]))
            # do projected GS sweeps
            for s in range(up):
                pgssweep(hierarchy[k],v,hierarchy[k].r,Psi,forward=False)
                if symmetric:
                    pgssweep(hierarchy[k],v,hierarchy[k].r,Psi)
            hierarchy[k].vstate += v
        else:
            hierarchy[k].vstate += hierarchy[k].P(hierarchy[k-1].vstate)

    # new iterate
    w += hierarchy[-1].vstate
    return w, chi

