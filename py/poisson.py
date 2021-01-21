# module for the linear Poisson equation, used in solving the classical obstacle problem

import numpy as np

__all__ = ['ellf','pointresidual','residual']

def ellf(mesh,f):
    '''Represent the linear functional (in (V^k)') which is the inner product
    with a function f(x):
       ell[v] = <f,v> = int_0^1 f(x) v(x) dx
    The values are  ell[p] = ell[psi_p^k]  for p=1,...,m_k, and
    ell[0]=ell[m+1]=0.  Uses trapezoid rule to evaluate the integrals
    ell[psi_p^k].  Input mesh is of class MeshLevel1D.'''
    assert len(f) == mesh.m+2, \
           'input vector f is of length %d (should be %d)' \
           % (len(f),mesh.m+2)
    ell = mesh.zeros()
    ell[1:-1] = mesh.h * f[1:-1]
    return ell

def pointresidual(mesh,w,ell,p):
    '''Compute the residual linear functional (in (V^k')) for given iterate w
    at one interior hat function psi_p^k:
       r(w)[psi_p^k] = ell(psi_p^k) - int_0^1 w'(x) (psi_p^k)'(x) dx
    Input ell is in (V^k)'.  Exactly computes the integral in a(.,.).
    Input mesh is of class MeshLevel1D.'''
    assert len(w) == mesh.m+2, \
           'input vector w is of length %d (should be %d)' \
           % (len(w),mesh.m+2)
    assert (p>=1) and (p<=mesh.m)
    return ell[p] - (1.0/mesh.h) * (2.0*w[p] - w[p-1] - w[p+1])

def residual(mesh,w,ell):
    '''Compute the residual linear functional (in (V^k')) for given iterate w:
       r(w)[v] = ell(v) - a(w,v)
               = ell(v) - int_0^1 w'(x) v'(x) dx
    The returned r = r(w) satisfies r[p] = r(w)[psi_p^k] and r[0]=r[m+1]=0.
    See above pointresidual() for further information.'''
    assert len(w) == mesh.m+2, \
           'input vector w is of length %d (should be %d)' \
           % (len(w),mesh.m+2)
    r = mesh.zeros()
    for p in range(1,mesh.m+1):
        r[p] = pointresidual(mesh,w,ell,p)
    return r

