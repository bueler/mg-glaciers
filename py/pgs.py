# module to implement the projected Gauss-Seidel (pGS) algorithm

from poisson import residual,formdiagonal,pointresidual
import numpy as np

__all__ = ['inactiveresidual','pgssweep']

def inactiveresidual(mesh,w,ell,phi,ireps=1.0e-10):
    '''Compute the values of the residual at nodes where the constraint
    is NOT active.  Note that where the constraint is active the residual
    may have significantly negative values.  The norm of the residual at
    inactive nodes is relevant to convergence.'''
    r = residual(mesh,w,ell)
    r[w < phi + ireps] = np.maximum(r[w < phi + ireps],0.0)
    return r

def pgssweep(mesh,w,ell,phi,forward=True,phieps=1.0e-10,printwarning=False):
    '''Do in-place projected Gauss-Seidel sweep, over the interior points
    p=1,...,m, for the classical obstacle problem
        -u'' - f >= 0,  u >= 0,  u (-u''-f) = 0
    Input iterate w is in V^k and ell is in (V^k)'.  At each p, solves
        r(w + c psi_p)[psi_p] = 0
    for c, thus c = r(w)[psi_p] / a(psi_p,psi_p) and then updates
        w[p] <- max(w[p]+c,phi[p])
    so w[p] >= phi[p].  See pointresidual(), formdiagonal() in poisson.py
    for r(w)[psi_p] and a(psi_p,psi_p).  Input mesh is of class MeshLevel1D.'''
    if forward:
        indices = range(1,mesh.m+1)
    else:
        indices = range(mesh.m,0,-1)
    for p in indices:
        if w[p] < phi[p] - phieps:
            if printwarning:
                print('WARNING: nonfeasible w[%d]=%e < phi[%d]=%e on level %d (m=%d)' \
                      % (p,w[p],p,phi[p],mesh.k,mesh.m))
            w[p] = phi[p]
        c = pointresidual(mesh,w,ell,p) / formdiagonal(mesh,p)
        w[p] = max(w[p]+c,phi[p])   # equivalent:  w[p] += max(c,phi[p]-w[p])
    return w

