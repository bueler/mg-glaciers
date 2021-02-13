'''Module to implement the projected Gauss-Seidel (pGS) algorithm.'''

import numpy as np
from poisson import residual, diagonalentry, pointresidual

__all__ = ['inactiveresidual', 'pgssweep']

def inactiveresidual(mesh, w, ell, phi, ireps=1.0e-10):
    '''Compute the values of the residual at nodes where the constraint
    is NOT active.  Note that where the constraint is active the residual
    may have significantly negative values.  The norm of the residual at
    inactive nodes is relevant to convergence.'''
    r = residual(mesh, w, ell)
    r[w < phi + ireps] = np.maximum(r[w < phi + ireps], 0.0)
    return r

def pgssweep(mesh, w, ell, phi, forward=True, phieps=1.0e-10,
             printwarnings=False):
    '''Do in-place projected Gauss-Seidel sweep, over the interior points
    p=1,...,m, for the classical obstacle problem
        u - phi >= 0,  -u'' - f >= 0,  (u - phi) (-u''-f) = 0
    Input iterate w is in V^j and ell is in V^j'.  At each p, solves
        r(w + c psi_p)[psi_p] = 0
    for c, where r(w)[v] = ell[v] - a(w,v).  Thus
        c = r(w)[psi_p] / a(psi_p,psi_p).
    The update is
        w[p] <- max(w[p]+c,phi[p])
    so w[p] >= phi[p].  Functions pointresidual() and diagonalentry()
    in poisson.py evaluate r(w)[psi_p] and a(psi_p,psi_p), respectively.
    Input mesh is of class MeshLevel1D.  Returns the number of pointwise
    feasibility violations.'''
    if forward:
        indices = range(1, mesh.m+1)
    else:
        indices = range(mesh.m, 0, -1)
    infeascount = 0
    for p in indices:
        if w[p] < phi[p] - phieps:
            if printwarnings:
                print('WARNING: nonfeasible w[%d]=%e < phi[%d]=%e on level %d (m=%d)' \
                      % (p, w[p], p, phi[p], mesh.j, mesh.m))
            w[p] = phi[p]
            infeascount += 1
        c = pointresidual(mesh, w, ell, p) / diagonalentry(mesh, p)
        w[p] = max(w[p] + c, phi[p])   # equivalent:  w[p] += max(c,phi[p]-w[p])
    mesh.WU += 1
    return infeascount
