'''Module for the linear Poisson equation, suitable for an obstacle problem.'''

__all__ = ['formdiagonal', 'pointresidual', 'residual']

def formdiagonal(mesh, p):
    '''Compute the diagonal of a(.,.) at one interior hat function psi_p^k:
       a(psi_p,psi_p) = int_0^1 (psi_p^k)'(x)^2 dx
    Input mesh is of class MeshLevel1D.'''
    assert 1 <= p <= mesh.m
    return 2.0 / mesh.h

def pointresidual(mesh, w, ell, p):
    '''Compute the residual linear functional (in (V^k')) for given iterate w
    at one interior hat function psi_p^k:
       r(w)[psi_p^k] = ell(psi_p^k) - int_0^1 w'(x) (psi_p^k)'(x) dx
    Input ell is in (V^k)'.  Exactly computes the integral in a(.,.).
    Input mesh is of class MeshLevel1D.'''
    assert len(w) == mesh.m+2, \
           'input vector w is of length %d (should be %d)' \
           % (len(w), mesh.m+2)
    assert 1 <= p <= mesh.m
    return ell[p] - (1.0/mesh.h) * (2.0*w[p] - w[p-1] - w[p+1])

def residual(mesh, w, ell):
    '''Compute the residual linear functional (in (V^k')) for given iterate w:
       r(w)[v] = ell(v) - a(w,v)
               = ell(v) - int_0^1 w'(x) v'(x) dx
    The returned r = r(w) satisfies r[p] = r(w)[psi_p^k] and r[0]=r[m+1]=0.
    See above pointresidual() for further information.'''
    assert len(w) == mesh.m+2, \
           'input vector w is of length %d (should be %d)' \
           % (len(w), mesh.m+2)
    r = mesh.zeros()
    for p in range(1, mesh.m+1):
        r[p] = pointresidual(mesh, w, ell, p)
    return r
