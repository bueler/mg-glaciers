'''Module for the linear Poisson equation -u''=f with u(0)=u(1)=0.'''

__all__ = ['diagonalentry', 'pointresidual', 'residual']

def _checklen(mesh, w, name):
    assert len(w) == mesh.m+2, \
           'input vector %s is of length %d (should be %d)' \
           % (name,len(w), mesh.m+2)

def diagonalentry(mesh, p):
    '''Compute the diagonal value of a(.,.) at hat function psi_p^j:
       a(psi_p,psi_p) = int_0^1 (psi_p^j)'(x)^2 dx
    Input mesh is of class MeshLevel1D.'''
    assert 1 <= p <= mesh.m
    return 2.0 / mesh.h

def pointresidual(mesh, w, ell, p):
    '''Compute the value of the residual linear functional, in V^j', for given
    iterate w, at one interior hat function psi_p^j:
       F(w)[psi_p^j] = ell(psi_p^j) - int_0^1 w'(x) (psi_p^j)'(x) dx
    Input ell is in V^j'.  Input mesh is of class MeshLevel1D.'''
    _checklen(mesh,w,'w')
    _checklen(mesh,ell,'ell')
    assert 1 <= p <= mesh.m
    return ell[p] - (1.0/mesh.h) * (2.0*w[p] - w[p-1] - w[p+1])

def residual(mesh, w, ell):
    '''Compute the residual linear functional, in V^j', for given iterate w:
       F(w)[v] = ell(v) - int_0^1 w'(x) v'(x) dx
    The returned F = F(w) satisfies F[p] = F(w)[psi_p^j] and F[0]=F[m+1]=0.
    See above pointresidual().'''
    _checklen(mesh,w,'w')
    _checklen(mesh,ell,'ell')
    F = mesh.zeros()
    for p in range(1, mesh.m+1):
        F[p] = pointresidual(mesh, w, ell, p)
    return F
