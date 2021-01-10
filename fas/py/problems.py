# module for the Problem1D class

import numpy as np

__all__ = ['LiouvilleBratu1D']

class Problem1D(object):
    '''Base class for 1D ordinary differential equation boundary value
    problems.'''

    def __init__(self):
        pass

    def F(self,h,w):
        return None

    def ngspoint(self,h,w,ell,p,niters=2):
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
    for all test functions v.'''

    def __init__(self,lam):
        self.lam = lam

    def F(self,h,w):
        '''Evaluate the weak form of the nonlinear operator
            F(w) = -w'' - lambda e^w,
        i.e.
            F(w)[v] = int_0^1 w'(x) v'(x) + lam e^{w(x)} v(x) dx
        for v equal to the interior-point hat functions psi_p at
        p=1,...,m-1.  Evaluates first integral exactly.  Last integral
        is by the trapezoid rule.  Input mesh is of class MeshLevel1D.
        Input w is a vector of length m+1.  The returned vector F is
        the same length as w and satisfies F[0]=F[m]=0.'''
        m = len(w) - 1
        FF = np.zeros(m+1)
        for p in range(1,m):
            FF[p] = (1.0/h) * (2.0*w[p] - w[p-1] - w[p+1]) \
                    - h * self.lam * np.exp(w[p])
        return FF

    def ngspoint(self,h,w,ell,p,niters=2):
        '''Do in-place nonlinear Gauss-Seidel (NGS) on vector w at an
        interior point p.  Uses a fixed number of Newton iterations on
            f(c) = r(w+c psi_p)[psi_p] = 0
        at point p, where  r(w)[v] = ell[v] - F(w)[v]  is the residual for w.
        The integral in F is computed by the trapezoid rule.  Newton steps
        are done without line search:
            c_{k+1} = c_k - f(c_k) / f'(c_k).'''
        c = 0
        for n in range(niters):
            tmp = h * self.lam * np.exp(w[p]+c)
            f = - (1.0/h) * (2.0*(w[p]+c) - w[p-1] - w[p+1]) + tmp + ell[p]
            df = - 2.0/h + tmp
            c -= f / df
        w[p] += c

    def mms(self,x):
        '''Return exact solution u(x) and right-hand-side g(x) for the
        method of manufactured solutions (MMS) case.'''
        u = np.sin(3.0 * np.pi * x)
        g = 9.0 * np.pi**2 * u - self.lam * np.exp(u)
        return u, g

