'''Module for the MeshLevel class suitable for obstacle problems.'''

import numpy as np

__all__ = ['MeshLevel1D']

class MeshLevel1D():
    '''Encapsulate a mesh level for the interval [0,xmax], suitable for
    obstacle problems.  MeshLevel1D(j=j) has m = 2^{j+1} - 1 interior nodes,
    m+1=2^{j+1} equal subintervals (elements) of length h = L / (m+1), and
    m+2 total points.  Indices j = 0,...,m+1 give all nodes, with
    j = 1,...,m giving the interior nodes:
        *---*---*---*---*---*---*---*
        0   1   2     ...  m-1  m  m+1
    Note MeshLevel1D(j=0) is a coarse mesh with one interior node and 2
    elements, MeshLevel1D(j=1) is a mesh with 3 interior nodes and 4 elements,
    and so on.  A MeshLevel1D(j) object knows about vectors in V^j including
    zero vectors, L_2 norms, linear functionals, canonical prolongation of
    functions (V^{j-1} to V^j), canonical restriction of linear functionals
    ((V^j)' to (V^{j-1})'), and monotone restriction of functions
    (V^j to V^{j-1}; see Graeser&Kornhuber 2009).'''

    def __init__(self, xmax=1, j=None):
        self.xmax = xmax
        self.j = j
        self.m = 2**(self.j+1) - 1
        if j > 0:
            self.mcoarser = 2**self.j - 1
        else:
            self.mcoarser = None
        self.h = self.xmax / (self.m + 1)
        self.WU = 0

    def checklen(self, v, coarser=False):
        '''Check whether the length of v matches the mesh.'''
        goodlen = self.mcoarser+2 if coarser else self.m+2
        assert len(v) == goodlen, \
               'input vector is of length %d (should be %d)' % (len(v), goodlen)

    def zeros(self):
        '''Allocate a zero vector.'''
        return np.zeros(self.m+2)

    def xx(self):
        '''Generate a vector of mesh node coordinates.'''
        return np.linspace(0.0, self.xmax, self.m+2)

    def l1norm(self, u):
        '''L^1[0,L] norm of a function, computed with trapezoid rule.'''
        self.checklen(u)
        return self.h * (0.5*abs(u[0]) + np.sum(abs(u[1:-1])) + 0.5*abs(u[-1]))

    def l2norm(self, u):
        '''L^2[0,L] norm of a function, computed with trapezoid rule.'''
        self.checklen(u)
        return np.sqrt(self.h * (0.5*u[0]*u[0] + np.sum(u[1:-1]*u[1:-1]) \
                                 + 0.5*u[-1]*u[-1]))

    def lqnorm(self, q, u):
        '''L^q[0,L] norm of a function, computed with trapezoid rule.'''
        self.checklen(u)
        assert q >= 1.0
        return (self.h * (0.5*u[0]**q + np.sum(u[1:-1]**q) + 0.5*u[-1]**q))**(1.0/q)

    def ellf(self, f):
        '''Represent the linear functional (in (V^j)') which is the inner
        product with a function f (in V^j):
           ellf[v] = <f,v> = int_0^L f(x) v(x) dx
        The values are  ellf[p] = ellf[psi_p^j]  for p=1,...,m_j, and
        ellf[0]=ellf[m+1]=0.  Uses trapezoid rule to evaluate the integrals
        ellf[psi_p^j].'''
        self.checklen(f)
        ellf = self.zeros()
        ellf[1:-1] = self.h * f[1:-1]
        return ellf

    def cP(self, v):
        '''Prolong a vector (function) on the next-coarser (j-1) mesh (i.e.
        in V^{j-1}) onto the current mesh (in S_j).  Uses linear interpolation.'''
        assert self.j > 0, \
               'cannot prolong from a mesh coarser than the coarsest mesh'
        self.checklen(v, coarser=True)
        y = self.zeros()  # y[0]=y[m+1]=0
        y[1] = 0.5 * v[1]
        for q in range(1, len(v)-2):
            y[2*q] = v[q]
            y[2*q+1] = 0.5 * (v[q] + v[q+1])
        y[-3] = v[-2]
        y[-2] = 0.5 * v[-2]
        return y

    def cR(self, ell):
        '''Restrict a linear functional ell on the current mesh (in (V^j)')
        to the next-coarser mesh, i.e. y = cR(ell) in (V^{j-1})', using
        "canonical restriction".'''
        assert self.j > 0, \
               'cannot restrict to a mesh coarser than the coarsest mesh'
        self.checklen(ell)
        y = np.zeros(self.mcoarser+2)  # y[0]=y[mcoarser+1]=0
        for q in range(1, self.mcoarser+1):
            y[q] = 0.5 * ell[2*q-1] + ell[2*q] + 0.5 * ell[2*q+1]
        return y

    def iR(self, v):
        '''Restrict a vector (function) ell on the current mesh (in V^j)
        to the next-coarser mesh, i.e. y = cR(v) in V^{j-1}, using
        injection.'''
        assert self.j > 0, \
               'cannot restrict to a mesh coarser than the coarsest mesh'
        self.checklen(v)
        y = np.zeros(self.mcoarser+2)  # y[0]=y[mcoarser+1]=0
        for q in range(1, self.mcoarser+1):
            y[q] = v[2*q]
        return y

    def injectP(self, ell):
        '''Prolong a linear functional ell by injection, from the next-coarser
        mesh.  If ell is in (V^{j-1})' then y = injectP(ell) is in (V^j)'.
        (Dual prolongation is an underdetermined problem, but injection is a
        solution.)'''
        assert self.j > 0, \
               'cannot prolong from a mesh coarser than the coarsest mesh'
        self.checklen(ell, coarser=True)
        y = self.zeros()
        for q in range(1, self.mcoarser+1):
            y[2*q] = ell[q]
        return y

    def mR(self, v):
        '''Evaluate the monotone restriction operator on a vector v
        on the current mesh (in V^j) to give a vector y = mR(v) on the
        next-coarser mesh (in V^{j-1}).  See formula (4.22) in G&K(2009).
        This is a nonlinear operation.'''
        assert self.j > 0, \
               'cannot restrict to a mesh coarser than the coarsest mesh'
        self.checklen(v)
        y = np.zeros(self.mcoarser+2)
        for q in range(1, self.mcoarser+1):
            y[q] = max(v[2*q-1:2*q+2])
        return y
