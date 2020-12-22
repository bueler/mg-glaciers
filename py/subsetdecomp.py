# module to implement the V-cycle algorithm for Tai (2003)
# multilevel subset decomposition method

import numpy as np

__all__ = ['pgssweep,vcycle']

def _fzero(x):
    return np.zeros(np.shape(x))

def _indentprint(n,s):
    '''Print 2n spaces and then string s.'''
    for i in range(n):
        print('  ',end='')
    print(s)

def _levelreport(indent,k,N,sweeps):
    _indentprint(indent-k,'mesh %d: %d sweeps over %d nodes' \
                          % (k,sweeps,N))

def _coarsereport(indent,N,sweeps):
    _indentprint(indent,'coarse: %d sweeps over %d nodes' \
                        % (sweeps,N))

def pgssweep(m,h,v,r,phi,backward=False):
    '''Do projected Gauss-Seidel sweep over the interior points p=1,...,m-1.
    At each p, solves
        a(v+c lam_p,lam_p) = r(lam_p)
    for c and then updates
        v[p] = max(v[p]+c,phi[p])
    so v[p] >= phi[p].'''
    indices = range(m-1,0,-1) if backward else range(1,m)
    for p in indices:
        # solve for c:  - (1/h) (v[p-1] - 2(v[p] + c) + v[p+1]) = r[p]
        c = 0.5 * (h*r[p] + v[p-1] + v[p+1]) - v[p]
        # enforce that  v_new[p] = v[p] + c  is above phi[p]
        v[p] = max(v[p]+c,phi[p])   # equivalent:  v[p] += max(c,phi[p] - v[p])
    return v

def vcycle(u,phi,f,meshes,levels=None,view=False,
           downsweeps=1,coarsesweeps=1,symmetric=False):
    '''Apply one V-cycle of Algorithm 4.7 in Graeser & Kornhuber (2009),
    namely the multilevel subset decomposition multigrid method from
    Tai (2003).  Updates (in place) the iterate u in K in
        a(u,v-u) >= (f,v-u)  for all v in K
    where K = {v >= phi}.  Vectors u,phi must be defined on the finest
    mesh meshes[levels-1] while input f is a function.  Note
    meshes[0],...,meshes[levels-1] (coarse to fine) are of type MeshLevel.
    The smoother is downsweeps (or upsweeps) iterations of projected
    Gauss-Seidel (pGS).  The coarse solver is coarsesweeps iterations
    of pGS, and thus not exact.'''
    # FIXME better symmetric smoother application: forward GS on down
    #                                              backward GS on up
    chi = [None] * (levels)           # empty list of length levels
    fine = levels - 1
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
    for k in range(1,fine+1):        # k=1,2,...,fine
        if view:
            _levelreport(fine,k,meshes[k].m-1,0)
        meshes[k].vstate += meshes[k].prolong(meshes[k-1].vstate)
    u += meshes[fine].vstate
    return u, chi

