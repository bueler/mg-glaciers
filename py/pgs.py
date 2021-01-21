# module to implement the projected Gauss-Seidel (pGS) algorithm

from poisson import formdiagonal, pointresidual
import numpy as np

__all__ = ['pgssweep']

def pgssweep(mesh,w,ell,phi,forward=True):
    '''Do projected Gauss-Seidel sweep, over the interior points p=1,...,m,
    for the Poisson equation.  At each p, solves
        r(w + c psi_p)[psi_p] = 0
    for c, thus c = r(w)[psi_p] / a(psi_p,psi_p) and then updates
        w[p] <- max(w[p]+c,phi[p])
    so w[p] >= phi[p].  Input iterate w is in V^k and ell is in (V^k)'.
    Input mesh is of class MeshLevel1D.'''
    if forward:
        indices = range(1,mesh.m+1)
    else:
        indices = range(mesh.m,0,-1)
    for p in indices:
        c = pointresidual(mesh,w,ell,p) / formdiagonal(mesh,p)
        w[p] = max(w[p]+c,phi[p])   # equivalent:  w[p] += max(c,phi[p]-w[p])
    return w

