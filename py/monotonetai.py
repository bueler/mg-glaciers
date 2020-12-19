# module to implement the V-cycle algorithm for 

import numpy as np

__all__ = ['vcycle']

def _fzero(x):
    '''The zero function.'''
    return np.zeros(np.shape(x))

def _indentprint(n,s):
    '''Indent n levels and print string s.'''
    for i in range(n):
        print('  ',end='')
    print(s)

# FIXME add coarsesweeps and upsweeps
def vcycle(hierarchy,phifine,uinitial,
           sweeps=None,view=False,level=None):
    '''One V-cycle of Algorithm 4.7 in Graeser & Kornhuber (2009).
    This monotone multigrid method is from Tai (2003).'''
    # SETUP
    mesh = hierarchy[-1]             # fine mesh
    chi = [None] * (level+1)         # empty list of length level+1
    chi[level] = phifine - uinitial  # fine mesh defect obstacle
    uu = uinitial.copy()
    r = mesh.residual(uu)            # fine mesh residual
    # DOWN-SMOOTH
    for k in range(level,0,-1):
        # monotone restriction gives defect obstacle (G&K formulas after (4.22))
        chi[k-1] = hierarchy[k].MRO(chi[k])
        # defect obstacle change on mesh k
        psi = chi[k] - hierarchy[k].prolong(chi[k-1])
        # do projected GS sweeps
        hk = hierarchy[k].h
        v = hierarchy[k].zeros()
        if view:
            _indentprint(level-k,'mesh %d: %d sweeps over %d interior points' \
                                 % (k,sweeps,hierarchy[k].m-1))
        # FIXME try symmetric
        for s in range(sweeps):
            hierarchy[k].pgssweep(v,r=r,phi=psi)
        hierarchy[k].vstate = v.copy()
        # update the residual and canonically-restrict it
        r += hierarchy[k].residual(v,f=_fzero)
        r = hierarchy[k].CR(r)
    # COARSE SOLVE using fixed number of projected GS sweeps
    psi = chi[0]
    h0 = hierarchy[0].h
    v = hierarchy[0].zeros()
    if view:
        _indentprint(level,'coarse: %d sweeps over %d interior points' \
                           % (sweeps,hierarchy[0].m-1))
    for s in range(sweeps):
        hierarchy[0].pgssweep(v,r=r,phi=psi)
    hierarchy[0].vstate = v.copy()
    # UP (WITHOUT SMOOTHING)
    # FIXME allow up-smoothing
    for k in range(1,level+1):
        if view:
            _indentprint(level-k,'mesh %d: interpolate up to %d interior points' \
                                 % (k,hierarchy[k].m-1))
        hierarchy[k].vstate += hierarchy[k].prolong(hierarchy[k-1].vstate)
    # FINALIZE
    uu += mesh.vstate
    return uu

