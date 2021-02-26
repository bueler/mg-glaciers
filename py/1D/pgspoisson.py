'''Module to implement the projected Gauss-Seidel (PGS) algorithm
for the linear Poisson equation -u''=f with u(0)=u(1)=0.'''

__all__ = ['pointresidual','residual','pgssweep']

def diagonalentry(mesh, p):
    '''Compute the diagonal value of a(.,.) at hat function psi_p^j:
       a(psi_p,psi_p) = int_0^1 (psi_p^j)'(x)^2 dx
    Input mesh is of class MeshLevel1D.'''
    assert 1 <= p <= mesh.m
    return 2.0 / mesh.h

def pointresidual(mesh, w, ell, p):
    '''Compute the value of the residual linear functional, in V^j', for given
    iterate w, at one interior hat function psi_p^j:
       F(w)[psi_p^j] = int_0^1 w'(x) (psi_p^j)'(x) dx - ell(psi_p^j)
    Input ell is in V^j'.  Input mesh is of class MeshLevel1D.'''
    mesh.checklen(w)
    mesh.checklen(ell)
    assert 1 <= p <= mesh.m
    return (1.0/mesh.h) * (2.0*w[p] - w[p-1] - w[p+1]) - ell[p]

def residual(mesh, w, ell):
    '''Compute the residual linear functional, in V^j', for given iterate w:
       F(w)[v] = int_0^1 w'(x) v'(x) dx - ell(v)
    The returned F = F(w) satisfies F[p] = F(w)[psi_p^j] and F[0]=F[m+1]=0.
    See above pointresidual().'''
    mesh.checklen(w)
    mesh.checklen(ell)
    F = mesh.zeros()
    for p in range(1, mesh.m+1):
        F[p] = pointresidual(mesh, w, ell, p)
    return F

def pgssweep(mesh, w, ell, phi, forward=True, omega=1.0, phieps=1.0e-10,
             printwarnings=False):
    '''Do in-place projected Gauss-Seidel sweep, with relaxation factor
    omega, over the interior points p=1,...,m, for the classical obstacle
    problem
        F(u)[v-u] = a(w,v-u) - ell[v-u] >= 0
    for all v in V^j.  Input iterate w is in V^j and ell is in V^j'.
    At each p, solves
        F(w + c psi_p)[psi_p] = 0
    for c.  Thus c = - F(w)[psi_p] / a(psi_p,psi_p).  Update of w guarantees
    admissibility:
        w[p] <- max(w[p] + omega c, phi[p]).
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
        w[p] = max(w[p] + omega * c, phi[p])
    mesh.WU += 1
    return infeascount
