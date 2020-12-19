# module for the MeshLevel class

import numpy as np

__all__ = ['MeshLevel']

class MeshLevel(object):
    '''Encapsulate a mesh level for the interval [0,1].  MeshLevel(k=0)
    is the coarse mesh and MeshLevel(k=j) is the fine mesh.
    MeshLevel(k=k) has m = 2^{k+1} subintervals.  This object knows
    about zero vectors, L_2 norms, prolongation,
    canonical restriction, monotone restriction, residuals, and
    projected Gauss-Seidel sweeps.'''

    def __init__(self, k=None, f=None):
        self.k = k
        self.f = f
        self.m = 2**(self.k+1)
        self.mcoarser = 2**self.k
        self.h = 1.0 / self.m
        self.xx = np.linspace(0.0,1.0,self.m+1)  # indices 0,1,...,m
        self.vstate = None

    def zeros(self):
        return np.zeros(self.m+1)

    def l2norm(self, u):
        '''L^2[0,1] norm computed with trapezoid rule.'''
        return np.sqrt(self.h * (0.5*u[0]*u[0] + np.sum(u[1:-1]*u[1:-1]) \
                                 + 0.5*u[-1]*u[-1]))

    def prolong(self,v):
        '''Prolong a vector on the next-coarser (k-1) mesh (i.e.
        in S_{k-1}) onto the current mesh (in S_k).'''
        assert len(v) == self.mcoarser+1, \
               'input vector of length %d (should be %d)' \
               % (len(v),self.mcoarser+1)
        assert self.k > 0, \
               'cannot prolong from a mesh coarser than the coarsest mesh'
        y = self.zeros()
        for q in range(self.mcoarser):
            y[2*q] = v[q]
            y[2*q+1] = 0.5 * (v[q] + v[q+1])
        y[-1] = v[-1]
        return y

    def CR(self,v):
        '''Restrict a linear functional (i.e. v in S_k') on the current mesh
        to the next-coarser (k-1) mesh using "canonical restriction".
        Only the interior points are updated.'''
        assert len(v) == self.m+1, \
               'input vector of length %d (should be %d)' % (len(v),self.m+1)
        assert self.k > 0, \
               'cannot restrict to a mesh coarser than the coarsest mesh'
        y = np.zeros(self.mcoarser+1)
        for q in range(1,len(y)-1):
            y[q] = 0.5 * (v[2*q-1] + v[2*q+1]) + v[2*q]
        return y

    def MR(self,v):
        '''Evaluate the monotone restriction operator on a vector v
        on the current mesh (i.e. v in S_k):
          y = R_k^{k-1} v.
        The result y is on the next-coarser (k-1) mesh, i.e. S_{k-1}.
        See formula (4.22) in G&K(2009).'''
        assert len(v) == self.m+1, \
               'input vector of length %d (should be %d)' % (len(v),self.m+1)
        assert self.k > 0, \
               'cannot restrict to a mesh coarser than the coarsest mesh'
        y = np.zeros(self.mcoarser+1)
        y[0] = max(v[0:2])
        for q in range(1,len(y)-1):
            y[q] = max(v[2*q-1:2*q+2])
        y[-1] = max(v[-2:])
        return y

    def residual(self,u,f=None):
        '''Represent the residual linear functional (i.e. in S_k')
           r(v) = ell(v) - a(u,v)
                = int_0^1 f v - int_0^1 u' v'
        associated to state u by a vector r for the interior points.
        Returned r satisfies r[0]=0 and r[m]=0.  Uses midpoint rule
        for the first integral and the exact value for the second.'''
        assert len(u) == self.m+1, \
               'input vector of length %d (should be %d)' % (len(v),self.m+1)
        if not f:
            f = self.f
        r = self.zeros()
        for p in range(1,self.m):
            xpm, xpp = (p-0.5) * self.h, (p+0.5) * self.h
            r[p] = (self.h/2.0) * (f(xpm) + f(xpp)) \
                   - (1.0/self.h) * (2.0*u[p] - u[p-1] - u[p+1])
        return r

    def pgssweep(self,v,r=None,phi=None):
        # FIXME option choosing forward, backward, symmetric
        for p in range(1,self.m):
            c = 0.5 * (self.h*r[p] + v[p-1] + v[p+1]) - v[p]
            v[p] += max(c,phi[p] - v[p])
        return v

