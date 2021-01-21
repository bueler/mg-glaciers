# module for the MeshLevel class suitable for obstacle problems

#TODO
#  1 separate PDE-specific computations residual(), inactiveresidual()
#  2 remove VR and VR0 methods?

import numpy as np

__all__ = ['MeshLevel1D']

class MeshLevel1D(object):
    '''Encapsulate a mesh level for the interval [0,1], suitable for
    obstacle problems.  MeshLevel1D(k=k) has m = 2^{k+1} - 1 interior nodes,
    m+1=2^{k+1} equal subintervals (elements) of length h = 1/(m+1), and
    m+2 total points.  Indices j = 0,...,m+1 give all nodes, with
    j = 1,...,m giving the interior nodes:
        *---*---*---*---*---*---*---*
        0   1   2     ...  m-1  m  m+1
    Note MeshLevel1D(k=0) is a coarse mesh with one interior node and 2
    elements, MeshLevel1D(k=1) is a mesh with 3 interior nodes and 4 elements,
    and so on.  A MeshLevel1D(k) object knows about vectors in V^k including
    zero vectors, L_2 norms, canonical prolongation of functions
    (V^{k-1} to V^k), canonical restriction of linear functionals
    ((V^k)' to (V^{k-1})'), and monotone restriction of functions
    (V^k to V^{k-1}; see Graeser&Kornhuber 2009).'''

    def __init__(self, k=None):
        self.k = k
        self.m = 2**(self.k+1) - 1
        if k > 0:
            self.mcoarser = 2**self.k - 1
        else:
            self.mcoarser = None
        self.h = 1.0 / (self.m + 1)
        self.vstate = None  #FIXME

    def zeros(self):
        return np.zeros(self.m+2)

    def xx(self):
        return np.linspace(0.0,1.0,self.m+2)

    def l2norm(self, u):
        '''L^2[0,1] norm of a function, computed with trapezoid rule.'''
        return np.sqrt(self.h * (0.5*u[0]*u[0] + np.sum(u[1:-1]*u[1:-1]) \
                                 + 0.5*u[-1]*u[-1]))

    def P(self,v):
        '''Prolong a vector (function) onto the next-coarser (k-1) mesh (i.e.
        in V^{k-1}) onto the current mesh (in S_k).  Uses linear interpolation.'''
        assert self.k > 0, \
               'cannot prolong from a mesh coarser than the coarsest mesh'
        assert len(v) == self.mcoarser+2, \
               'input vector v is of length %d (should be %d)' \
               % (len(v),self.mcoarser+2)
        y = self.zeros()  # y[0]=y[m+1]=0
        y[1] = 0.5 * v[1]
        for q in range(1,len(v)-2):
            y[2*q] = v[q]
            y[2*q+1] = 0.5 * (v[q] + v[q+1])
        y[-3] = v[-2]
        y[-2] = 0.5 * v[-2]
        return y

    def cR(self,ell):
        '''Restrict a linear functional ell on the current mesh (in (V^k)')
        to the next-coarser mesh, i.e. y = cR(ell) in (V^{k-1})', using
        "canonical restriction".'''
        assert self.k > 0, \
               'cannot restrict to a mesh coarser than the coarsest mesh'
        assert len(ell) == self.m+2, \
               'input vector v is of length %d (should be %d)' \
               % (len(ell),self.m+2)
        y = np.zeros(self.mcoarser+2)  # y[0]=y[mcoarser+1]=0
        for q in range(1,self.mcoarser+1):
            y[q] = 0.5 * ell[2*q-1] + ell[2*q] + 0.5 * ell[2*q+1]
        return y

    def mR(self,v):
        '''Evaluate the monotone restriction operator on a vector v
        on the current mesh (in V^k) to give a vector y = mR(v) on the
        next-coarser mesh (in V^{k-1}).  See formula (4.22) in G&K(2009).'''
        assert self.k > 0, \
               'cannot restrict to a mesh coarser than the coarsest mesh'
        assert len(v) == self.m+2, \
               'input vector v is of length %d (should be %d)' \
               % (len(v),self.m+2)
        y = np.zeros(self.mcoarser+2)
        for q in range(1,self.mcoarser+1):
            y[q] = max(v[2*q-1:2*q+2])
        return y

    #FIXME remove?
    def VR0(self,v):
        '''Restrict a vector v in S_k on the current mesh to the next-coarser
        (k-1) mesh by using full-weighting.  Only the interior points are
        updated and the returned vector has zero boundary values.'''
        assert self.k > 0, \
               'cannot restrict to a mesh coarser than the coarsest mesh'
        assert len(v) == self.m+1, \
               'input vector v is of length %d (should be %d)' % (len(v),self.m+1)
        y = np.zeros(self.mcoarser+1)
        for q in range(1,len(y)-1):
            y[q] = 0.25 * (v[2*q-1] + v[2*q+1]) + 0.5 * v[2*q]
        return y

    #FIXME remove?
    def VR(self,v):
        '''Restrict a vector v in S_k on the current mesh to the next-coarser
        (k-1) mesh by using full-weighting.  All points are updated.  For
        boundary values see equation (6.20) in Bueler (2021).'''
        assert len(v) == self.m+1, \
               'input vector v is of length %d (should be %d)' % (len(v),self.m+1)
        assert self.k > 0, \
               'cannot restrict to a mesh coarser than the coarsest mesh'
        y = np.zeros(self.mcoarser+1)
        y[0] = (2.0/3.0) * v[0] + (1.0/3.0) * v[1]
        for q in range(1,len(y)-1):
            y[q] = 0.25 * (v[2*q-1] + v[2*q+1]) + 0.5 * v[2*q]
        y[-1] = (1.0/3.0) * v[-2] + (2.0/3.0) * v[-1]
        return y

    #FIXME move to obs1.py?
    def inactiveresidual(self,u,f,phi):
        '''Compute the values of the residual at nodes where the constraint
        is NOT active.  Note that where the constraint is active the residual
        may have significantly negative values.  The norm of the residual at
        inactive nodes is relevant to convergence.'''
        r = self.residual(u,f)
        osreps = 1.0e-10
        r[u < phi + osreps] = np.maximum(r[u < phi + osreps],0.0)
        return r

