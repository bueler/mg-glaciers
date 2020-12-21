# module to implement the V-cycle algorithm for 

import numpy as np

__all__ = ['pgssweep,vcycle']

def _fzero(x):
    return np.zeros(np.shape(x))

def _indentprint(n,s):
    for i in range(n):
        print('  ',end='')
    print(s)

def _levelreport(fine,k,N,sweeps):
    _indentprint(fine-k,'mesh %d: %d sweeps over %d nodes' \
                        % (k,sweeps,N))

def _coarsereport(fine,N,sweeps):
    _indentprint(fine,'coarse: %d sweeps over %d nodes' \
                      % (sweeps,N))

def pgssweep(m,h,v,r,phi,backward=False):
    indices = range(m-1,0,-1) if backward else range(1,m)
    for p in indices:
        c = 0.5 * (h*r[p] + v[p-1] + v[p+1]) - v[p]
        v[p] += max(c,phi[p] - v[p])
    return v

def vcycle(u,phi,f,meshes,fine=None,view=False,
           downsweeps=1,coarsesweeps=1,upsweeps=0,
           symmetric=False):
    '''Apply one V-cycle of Algorithm 4.7 in Graeser & Kornhuber (2009),
    namely the subset decomposition multigrid method from Tai (2003).
    Updates (in place) the iterate u in K in
        a(u,v-u) >= (f,v-u)  for all v in K
    where K = {v >= phi}.  Vectors u,phi must be defined on the finest
    mesh meshes[fine] while input f is a function.  Note
    meshes[0],...,meshes[fine] (coarse to fine) are of type MeshLevel.
    The smoother is downsweeps (or upsweeps) iterations of projected
    Gauss-Seidel (pGS).  The coarse solver is coarsesweeps iterations
    of pGS, and thus not exact.'''
    # FIXME better symmetric smoother application: forward GS on down
    #                                              backward GS on up
    if not fine:
        fine = len(meshes) - 1
    chi = [None] * (fine+1)           # empty list of length fine+1
    chi[fine] = phi - u               # fine mesh defect obstacle
    r = meshes[fine].residual(u,f)    # fine mesh residual
    # DOWN
    for k in range(fine,0,-1):        # k=fine,fine-1,...,1
        # monotone restriction gives defect obstacle
        chi[k-1] = meshes[k].MR(chi[k])
        # the actual obstacle is the *change* in chi
        psi = chi[k] - meshes[k].prolong(chi[k-1])
        # do projected GS sweeps
        if view:
            _levelreport(fine,k,meshes[k].m-1,downsweeps)
        v = meshes[k].zeros()
        for s in range(downsweeps):
            pgssweep(meshes[k].m,meshes[k].h,v,r,psi)
            if symmetric:
                pgssweep(meshes[k].m,meshes[k].h,v,r,psi,backward=True)
        meshes[k].vstate = v.copy()
        # update and canonically-restrict the residual
        r += meshes[k].residual(v,_fzero)
        r = meshes[k].CR(r)
    # COARSE SOLVE
    psi = chi[0]
    if view:
        _coarsereport(fine,meshes[0].m-1,coarsesweeps)
    v = meshes[0].zeros()
    for s in range(coarsesweeps):
        pgssweep(meshes[0].m,meshes[0].h,v,r,psi)
        if symmetric:
            pgssweep(meshes[0].m,meshes[0].h,v,r,psi,backward=True)
    meshes[0].vstate = v.copy()
    # UP
    assert (upsweeps==0), 'up sweeps not implemented yet' # FIXME
    for k in range(1,fine+1):        # k=1,2,...,fine
        if view:
            _levelreport(fine,k,meshes[k].m-1,upsweeps)
        meshes[k].vstate += meshes[k].prolong(meshes[k-1].vstate)
    u += meshes[fine].vstate
    return u

