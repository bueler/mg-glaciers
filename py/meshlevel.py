# module for the MeshLevel class suitable for obstacle problems

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
    zero vectors, L_2 norms, linear functionals, canonical prolongation of
    functions (V^{k-1} to V^k), canonical restriction of linear functionals
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

    def checklen(self,v,coarser=False):
        goodlen = self.mcoarser+2 if coarser else self.m+2
        assert len(v) == goodlen, \
               'input vector is of length %d (should be %d)' \
               % (len(v),goodlen)

    def zeros(self):
        return np.zeros(self.m+2)

    def xx(self):
        return np.linspace(0.0,1.0,self.m+2)

    def l2norm(self, u):
        '''L^2[0,1] norm of a function, computed with trapezoid rule.'''
        self.checklen(u)
        return np.sqrt(self.h * (0.5*u[0]*u[0] + np.sum(u[1:-1]*u[1:-1]) \
                                 + 0.5*u[-1]*u[-1]))

    def ell(self,f):
        '''Represent the linear functional (in (V^k)') which is the inner
        product with a function f (in V^k):
           ell[v] = <f,v> = int_0^1 f(x) v(x) dx
        The values are  ell[p] = ell[psi_p^k]  for p=1,...,m_k, and
        ell[0]=ell[m+1]=0.  Uses trapezoid rule to evaluate the integrals
        ell[psi_p^k].'''
        self.checklen(f)
        ell = self.zeros()
        ell[1:-1] = self.h * f[1:-1]
        return ell

    def P(self,v):
        '''Prolong a vector (function) onto the next-coarser (k-1) mesh (i.e.
        in V^{k-1}) onto the current mesh (in S_k).  Uses linear interpolation.'''
        assert self.k > 0, \
               'cannot prolong from a mesh coarser than the coarsest mesh'
        self.checklen(v,coarser=True)
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
        self.checklen(ell)
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
        self.checklen(v)
        y = np.zeros(self.mcoarser+2)
        for q in range(1,self.mcoarser+1):
            y[q] = max(v[2*q-1:2*q+2])
        return y

