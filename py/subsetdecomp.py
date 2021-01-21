# module to implement the V-cycle algorithm for Tai (2003)
# multilevel subset decomposition method

from pgs import pgssweep
import numpy as np

__all__ = ['vcycle']

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

def vcycle(u,phi,f,meshes,levels=None,view=False,symmetric=False,
           downsweeps=1,coarsesweeps=1,upsweeps=0):
    '''Apply one V(1,0)-cycle of the multilevel subset decomposition
    method from Tai (2003).  (Alg. 4.7 in Graeser & Kornhuber (2009).)
    In-place updates the iterate u in K = {v | v >= phi} in the VI
        a(u,v-u) >= (f,v-u)  for all v in K
    Vectors u,phi must be defined on the finest mesh (meshes[levels-1])
    while input f is a function.  Note meshes[0],...,meshes[levels-1]
    (coarse to fine) are of type MeshLevel.  The smoother is downsweeps
    iterations of projected Gauss-Seidel (pGS).  The coarse solver is
    coarsesweeps iterations of pGS, and thus not exact.'''
    chi = [None] * (levels)           # empty list of length levels
    fine = levels - 1
    chi[fine] = phi - u               # fine mesh defect obstacle
    r = meshes[fine].residual(u,f)    # fine mesh residual
    # DOWN
    for k in range(fine,0,-1):        # k=fine,fine-1,...,1
        # monotone restriction decomposes defect obstacle
        chi[k-1] = meshes[k].MR(chi[k])
        # the level k obstacle is the *change* in chi
        psi = chi[k] - meshes[k].prolong(chi[k-1])
        if upsweeps > 0:
            psi *= 0.5
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
    psi = chi[0]  # correct for any upsweeps
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
            _levelreport(fine,k,meshes[k].m-1,upsweeps)
        # FIXME something in the following is WRONG ... V(1,1) cycles do not work
        w = meshes[k].prolong(meshes[k-1].vstate)
        if upsweeps > 0:
            psi = 0.5 * (chi[k] - meshes[k].prolong(chi[k-1]))
            r = meshes[k].residual(w,_fzero)  # FIXME update stored level k residual?
            #v = meshes[k].zeros()
            for s in range(upsweeps):
                pgssweep(meshes[k].m,meshes[k].h,w,r,psi)
                if symmetric:
                    pgssweep(meshes[k].m,meshes[k].h,w,r,psi,backward=True)
            #meshes[k].vstate = v.copy()
        meshes[k].vstate += w
    u += meshes[fine].vstate
    return u, chi

