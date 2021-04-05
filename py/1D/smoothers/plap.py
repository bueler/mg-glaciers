'''Module for PNsmootherPLap classes, a derived class of PNsmootherSIA.
Provides GS and Jacobi smoothers, and an exact solution, for the p-Laplacian
obstacle problem.'''

__all__ = ['PNsmootherPLap']

import numpy as np
from smoothers.sia import PNsmootherSIA

class PNsmootherPLap(PNsmootherSIA):
    '''Smoother for the p-Laplacian obstacle problem with p=4:
        - (|u'(x)|^2 u'(x))' = f(x),  u(x) >= phi(x)
    with u(0)=u(1)=0.

    One "bridge" exact solution is implemented, symmetric around x=0.5:
        f(x) = 3 (x-0.5)^2
        phi(x) = 0.25 sin(3 pi x)
    (Actually we take phi(x) = max(0, 0.25 sin(3 pi x).)
    Integrating once using u'(0.5)=0 from symmetry, get
        - (u'(x))^3 = (x-0.5)^3
    so u'(x) = 0.5 - x and u(x) = 0.5 x (1 - x) + C, thus u(x) is a downward
    parabola at a height to be determined by the obstacle.  By sketching
    there is a unique 0 < a < 1/6 so that
        u(a) = phi(a)  and  u'(a) = phi'(a)
    The latter equation is
        F(a) = 0.5 - a - 0.75 pi cos(3 pi a) = 0
    which gives a = 0.1508874586825051 by Newton's method (see _solvefora()).
    Given a, the first equation says 0.5 a (1 - a) + C = 0.25 sin(3 pi a)
    so C = 0.18318033594477562.  See exact() and exactfigure().

    In other ways just a simplification of the SIA smoother PNsmootherSIA.'''

    def __init__(self, args, admissibleeps=1.0e-10):
        super().__init__(args, admissibleeps=admissibleeps)
        # parameter used in pointupdate()
        self.cupmax = 1.0  # never move surface up by more than this

    def _pointN(self, mesh, w, p):
        '''Compute nonlinear operator value N(w)[psi_p^j], for
        given iterate w(x) in V^j, at one hat function psi_p^j:
           N(w)[psi_p^j] = int_I |w'(x)|^2 w'(x) (psi_p^j)'(x) dx
        where I = [x_p - h, x_p + h].  Also return dNdw, the derivative
        of N(w)[psi_p^j] with respect to w[p].'''
        assert hasattr(mesh, 'h')
        dw = w[p:p+2] - w[p-1:p+1]
        N = (1.0 / mesh.h**3) * (dw[0]**3.0 - dw[1]**3.0)
        dNdw = (3.0 / mesh.h**3) * (dw[0]**2.0 + dw[1]**2.0)
        return N, dNdw

    def pointupdate(self, r, delta, yp, phip, ellp):
        '''Compute update value c, for y[p] += c, from pointwise residual
        r = rho(0) and pointwise Jacobian delta = rho'(0).'''
        if delta == 0.0:
            return 0.0                # no information on how to move
        else:
            c = - r / delta           # pure Newton step
            c = max(c, phip - yp)     # ensure admissibility: y >= phi
            c = min(c, self.cupmax)   # limit large upward steps
            return c

    def source(self, x):
        return 3.0 * (x - 0.5)**2

    def phi(self, x):
        '''Obstacle.'''
        #return np.maximum(0.0, 0.25 * np.sin(3.0 * np.pi * x))
        return 0.25 * np.sin(3.0 * np.pi * x)

    def initial(self, x):
        '''Default initial shape.'''
        return np.maximum(self.phi(x), 0.0)

    def exact_available(self):
        return True

    def _solvefora(self):
        '''Find constant in exact solution by root finding on system
            u(a) = phi(a), u'(a) = phi'(a)
        The solution is in (0,1/6) and close to 0.15.'''
        def ff(a):
            return 0.5 - a - 0.75 * np.pi * np.cos(3.0 * np.pi * a)
        def dff(a):
            return -1.0 + 2.25 * np.pi**2 * np.sin(3.0 * np.pi * a)
        a = 0.15
        print(a)
        for k in range(3):
            a = a - ff(a) / dff(a)
            print(a)
        return a

    def exact(self, x):
        '''Exact solution.'''
        a = 0.1508874586825051
        C = 0.18318033594477562
        u = self.phi(x)
        mid = (x > a) * (x < 1.0 - a)
        u[mid] = 0.5 * x[mid] * (1.0 - x[mid]) + C
        return u

    def exactfigure(self, mesh):
        '''Show exact solution in a basic figure.'''
        x = mesh.xx()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16.0, 5.0))
        plt.plot(x, self.phi(x), 'k', label='obstacle')
        plt.plot(x, self.exact(x), 'k--', label='$u_{exact}$')
        plt.legend(fontsize=12.0)
        plt.xlabel('x', fontsize=12.0)
        plt.savefig('plapfigure.pdf', bbox_inches='tight')
