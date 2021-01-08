# module for the Problem1D class

import numpy as np

__all__ = ['LiouvilleBratu1D']

class Problem1D(object):
    '''Base class for 1D ordinary differential equation boundary value
    problems.'''

    def __init__(self):
        pass

    def F(self,mesh,w):
        return None

    def ngssweep(self,mesh,w,ell,forward=True,niters=2):
        return None

    def mms(self,x):
        return None

class LiouvilleBratu1D(Problem1D):
    '''Evaluate the weak-form operator F() and NGS sweeps ngssweeps() for
    the nonlinear Liouville-Bratu problem
        -u'' - lambda e^u = g,  u(0) = u(1) = 0
    where lambda is constant.  In fact g(x) on the right is replaced by
    a linear functional, so the problem is
        F(w)[v] = ell[v]
    for all test functions v.  A nontrivial g(x) is computed by mms().'''

    def __init__(self,lam):
        self.lam = lam

    def F(self,mesh,w):
        '''Evaluate the weak form of the nonlinear operator
            F(w) = -w'' - lambda e^w,
        i.e.
            F(w)[v] = int_0^1 w'(x) v'(x) + lam e^{w(x)} v(x) dx
        for v equal to the interior-point hat functions psi_p at
        p=1,...,m-1.  Evaluates first integral exactly.  Last integral
        is by the trapezoid rule.  Input mesh is of class MeshLevel1D.
        Input w is a vectors of length m+1.  The returned vector F is
        of length m+1 and satisfies F[0]=F[m]=0.'''
        assert len(w) == mesh.m+1, \
              'input vector u is of length %d (should be %d)' \
              % (len(w),mesh.m+1)
        FF = mesh.zeros()
        for p in range(1,mesh.m):
            FF[p] = (1.0/mesh.h) * (2.0*w[p] - w[p-1] - w[p+1]) \
                   - mesh.h * self.lam * np.exp(w[p])
        return FF

    def ngssweep(self,mesh,w,ell,forward=True,niters=2):
        '''Do one in-place nonlinear Gauss-Seidel (NGS) sweep on vector w
        over the interior points p=1,...,m-1.  Use a fixed number of
        Newton iterations on
            f(c) = 0
        at each point, i.e. with
            f(c) = r(w+c psi_p)[psi_p]
        where  r(w)[v] = ell[v] - F(w)[v]  is the residual for w.  The
        integral in F is computed by the trapezoid rule.  Newton steps are
        without line search:
            f'(c_k) s_k = - f(c_k)
            c_{k+1} = c_k + s_k.'''
        if forward:
            indices = range(1,mesh.m)
        else:
            indices = range(mesh.m-1,0,-1)
        for p in indices:
            c = 0   # because previous iterate w is close to correct
            for n in range(niters):
                tmp = mesh.h * self.lam * np.exp(w[p]+c)
                f = - (1.0/mesh.h) * (2.0*(w[p]+c) - w[p-1] - w[p+1]) \
                    + tmp + ell[p]
                df = - 2.0/mesh.h + tmp
                c -= f / df
            w[p] += c
        return w

    def mms(self,x):
        u = np.sin(3.0 * np.pi * x)
        g = 9.0 * np.pi**2 * u - self.lam * np.exp(u)
        return u, g

