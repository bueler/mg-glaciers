'''Module to implement the projected Gauss-Seidel (pGS) algorithm.'''

import numpy as np
from poisson import residual, diagonalentry, pointresidual

__all__ = ['inactiveresidual', 'pgssweep']

def inactiveresidual(mesh, w, ell, phi, ireps=1.0e-10):
    '''Compute the values of the residual for w at nodes where the constraint
    is NOT active.  Note that where the constraint is active the residual F(w)
    in the complementarity problem is allowed to have any positive value, and
    only the residual at inactive nodes is relevant to convergence.'''
    F = residual(mesh, w, ell)
    F[w < phi + ireps] = np.minimum(F[w < phi + ireps], 0.0)
    return F

def pgssweep(mesh, w, ell, phi, forward=True, phieps=1.0e-10,
             printwarnings=False):
    '''Do in-place projected Gauss-Seidel sweep, over the interior points
    p=1,...,m, for the classical obstacle problem
        F(u)[v-u] = a(w,v-u) - <f,v-u> >= 0
    for all v in V^j.  Input iterate w is in V^j and ell is in V^j'.
    At each p, solves
        F(w + c psi_p)[psi_p] = 0
    for c, where F(w)[v] = a(w,v) - ell[v].  Thus
        c = - F(w)[psi_p] / a(psi_p,psi_p).
    The update of w guarantees admissibility (w[p] >= phi[p])
        w[p] <- max(w[p]+c,phi[p]).
    Functions pointresidual() and diagonalentry() in poisson.py evaluate
    F(w)[psi_p] and a(psi_p,psi_p), respectively.  Input mesh is of class
    MeshLevel1D.  Returns the number of pointwise feasibility violations.'''
    if forward:
        indices = range(1, mesh.m+1)    # 1,...,m
    else:
        indices = range(mesh.m, 0, -1)  # m,...,1
    infeascount = 0
    for p in indices:
        if w[p] < phi[p] - phieps:
            if printwarnings:
                print('WARNING: repairing nonfeasible w[%d]=%e < phi[%d]=%e on level %d (m=%d)' \
                      % (p, w[p], p, phi[p], mesh.j, mesh.m))
            w[p] = phi[p]
            infeascount += 1
        c = - pointresidual(mesh, w, ell, p) / diagonalentry(mesh, p)
        c = max(c,phi[p] - w[p])
        w[p] = w[p] + c
    mesh.WU += 1
    return infeascount
