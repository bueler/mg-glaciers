# module for the MeshLevel1D class

import numpy as np

__all__ = ['MeshLevel1D']

class MeshLevel1D(object):
    '''Encapsulate a mesh level for the interval [0,1], suitable for
    FAS.  MeshLevel(k=k) has m = 2^{k+1} equal subintervals
    of length h = 1/m.  Indices give nodes 0,1,...,m:
        *---*---*---*---*---*---*
        0   1   2     ...  m-1  m
    Note p=1,...,m-1 are interior nodes.  MeshLevel1D(k=0) is the coarse mesh
    with one interior node.  This object knows about zero vectors, L_2 norms,
    prolongation (k-1 to k), canonical restriction of linear functionals
    (k to k-1), and ordinary restriction of functions (k to k-1) by
    full-weighting.'''

    def __init__(self, k):
        self.k = k
        self.m = 2**(self.k+1)
        self.mcoarser = 2**self.k
        self.h = 1.0 / self.m

    def zeros(self):
        return np.zeros(self.m+1)

    def xx(self):
        return np.linspace(0.0,1.0,self.m+1)

    def l2norm(self, u):
        '''L^2[0,1] norm computed with trapezoid rule.'''
        return np.sqrt(self.h * (0.5*u[0]*u[0] + np.sum(u[1:-1]*u[1:-1]) \
                                 + 0.5*u[-1]*u[-1]))

    def prolong(self,v):
        '''Prolong a function on the next-coarser (k-1) mesh (i.e.
        in S_{k-1}) onto the current mesh (in S_k).  Uses linear
        interpolation.  Ignores the values v[0] and v[mcoarser]
        because we only prolong functions for which they are zero.'''
        assert len(v) == self.mcoarser+1, \
               'input vector v is of length %d (should be %d)' \
               % (len(v),self.mcoarser+1)
        assert self.k > 0, \
               'cannot prolong from a mesh coarser than the coarsest mesh'
        y = self.zeros()  # y[0]=y[m]=0
        y[1] = 0.5 * v[1]
        for q in range(1,self.mcoarser-1):
            y[2*q] = v[q]
            y[2*q+1] = 0.5 * (v[q] + v[q+1])
        y[self.m-2] = v[self.mcoarser-1]
        y[self.m-1] = 0.5 * v[self.mcoarser-1]
        return y

    def CR(self,v):
        '''Restrict a linear functional (i.e. v in S_k') on the current mesh
        to the next-coarser (k-1) mesh using "canonical restriction".
        Only the interior points are updated.'''
        assert len(v) == self.m+1, \
               'input vector v is of length %d (should be %d)' % (len(v),self.m+1)
        assert self.k > 0, \
               'cannot restrict to a mesh coarser than the coarsest mesh'
        y = np.zeros(self.mcoarser+1)  # y[0]=y[mcoarser]=0
        for q in range(1,self.mcoarser):
            y[q] = 0.5 * v[2*q-1] + v[2*q] + 0.5 * v[2*q+1]
        return y

    def Rfw(self,v):
        '''Restrict a vector (function) v in S_k on the current mesh to the
        next-coarser (k-1) mesh by using full-weighting.'''
        assert len(v) == self.m+1, \
               'input vector v is of length %d (should be %d)' % (len(v),self.m+1)
        assert self.k > 0, \
               'cannot restrict to a mesh coarser than the coarsest mesh'
        y = np.zeros(self.mcoarser+1)  # y[0]=y[mcoarser]=0
        for q in range(1,self.mcoarser):
            y[q] = 0.25 * v[2*q-1] + 0.5 * v[2*q] + 0.25 * v[2*q+1]
        return y

