'''Module for PNsmootherPLap classes, a derived class of PNsmootherSIA.
Provides GS and Jacobi smoothers, and two exact solutions, for the p-Laplacian
obstacle problem.'''

__all__ = ['PNsmootherPLap']

import numpy as np
from smoothers.sia import PNsmootherSIA

class PNsmootherPLap(PNsmootherSIA):
    '''Smoother for the p-Laplacian obstacle problem with p=4:
        - (|u'(x)|^2 u'(x))' = f(x),  u(x) >= phi(x)
    with u(0)=u(1)=0.  Two exact solutions are implemented, both symmetric
    around x=0.5:

    1. "pile":  Data
        f(x) = 1 on [0.3,0.7] and -1 on [0,0.3) or (0.7,1]
        phi(x) = 0
    We force symmetry around x=0.5 and free boundaries at x=0.1 and x=0.9.
    Integrating and using u'(0.5)=0 and u(0.5)=A get
        u(x) = A - (3/4) |x-0.5|^(4/3)  on [0.3,0.7]
    On [0.1,0.3] or [0.7,0.9] we have u'(x)^3 = - (0.4 - |x-0.5|) from the
    free boundary, thus
        u(x) = A - (3/2) (0.2)^(4/3) + (3/4) (0.4 - |x-0.5|)^(4/3)
    Use u(0.4) to find A = (3/2) (0.2)^(4/3) thus
        u(x) = (3/4) (0.4 - |x-0.5|)^(4/3)  on [0.1,0.3] or [0.7,0.9].
    Finally
        u(x)=0 on [0,0.1] or [0.9,1].

    2. "bridge":  Data
        f(x) = 3 (x-0.5)^2
        phi(x) = 0.25 sin(3 pi x)
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
        # parameter used in _pointupdate()
        self.cupmax = 1.0  # never move surface up by more than this
        # parameter used in gssweep()
        self.ctol = 0.001

    def _pointN(self, mesh, wpatch, p):
        '''Compute nonlinear operator value N(w)[psi_p^j], for
        given iterate w(x) in V^j, at one hat function psi_p^j:
           N(w)[psi_p^j] = int_I |w'(x)|^2 w'(x) (psi_p^j)'(x) dx
        where I = [x_p - h, x_p + h].  Also return dNdw, the derivative
        of N(w)[psi_p^j] with respect to w[p].  Note that
        wpatch = [w[p-1], w[p], w[p+1]].'''
        assert hasattr(mesh, 'h')
        assert len(wpatch) == 3
        dw = wpatch[1:] - wpatch[:2]
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
        '''Source f(x) in -(|u'|^2 u')' = f.'''
        if self.args.plapcase == 'pile':
            f = np.ones(np.shape(x))
            f[x < 0.3] = - 1.0
            f[x > 0.7] = - 1.0
            return f
        elif self.args.plapcase == 'bridge':
            f = 3.0 * (x - 0.5)**2
        return f

    def phi(self, x):
        '''Obstacle phi(x).'''
        if self.args.plapcase == 'pile':
            return np.zeros(np.shape(x))
        elif self.args.plapcase == 'bridge':
            return 0.25 * np.sin(3.0 * np.pi * x)

    def initial(self, x):
        '''Default initial shape.'''
        if self.args.plapcase == 'pile':
            return x * (1.0 - x)  # needs to be positive to have any movement
        elif self.args.plapcase == 'bridge':
            return np.maximum(self.phi(x), 0.0)

    def exact_available(self):
        return True

    def _solvefora(self):
        '''Find constant in "bridge" exact solution by root finding on system
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
        if self.args.plapcase == 'pile':
            u = np.zeros(np.shape(x))
            xam = abs(x - 0.5)
            mid = (xam <= 0.2)
            next = (xam > 0.2) * (xam < 0.4)
            r = 4.0 / 3.0
            u[mid] = 1.5 * 0.2**r - 0.75 * xam[mid]**r
            u[next] = 0.75 * (0.4 - xam[next])**r
            return u
        elif self.args.plapcase == 'bridge':
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
        fname = 'plapfigure.pdf'
        print('saving image of plap exact solution to %s ...' % fname)
        plt.savefig(fname, bbox_inches='tight')
