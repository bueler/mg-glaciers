'''Module for PNGSPLap and PNJacobiPLap classes, a derived class from
SmootherObstacleProblem.  Provides smoothers and exact solutions
for the p-Laplacian obstacle problem.'''

__all__ = ['PNGSPLap', 'PNJacobiPLap']

import numpy as np
from smoothers.base import SmootherObstacleProblem

class PNGSPLap(SmootherObstacleProblem):
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

    In other ways this example is just a simplification of the SIA smoothers
    PNGSSIA and PNJacobiSIA.'''

    def __init__(self, args, admissibleeps=1.0e-10):
        super().__init__(args, admissibleeps=admissibleeps)
        self.args = args
        # parameters used in PNGS and PNJacobi smoother
        self.newtonupwardmax = 1.0  # never move surface up by more than this

    def _pointN(self, h, w, p):
        '''Compute nonlinear operator value N(w)[psi_p^j], for
        given iterate w(x) in V^j, at one hat function psi_p^j:
           N(w)[psi_p^j] = int_I |w'(x)|^2 w'(x) (psi_p^j)'(x) dx
        where I = [x_p - h, x_p + h].  Also return dNdw, the derivative
        of N(w)[psi_p^j] with respect to w[p].'''
        dw = w[p:p+2] - w[p-1:p+1]
        N = (1.0 / h**3) * (dw[0]**3.0 - dw[1]**3.0)
        dNdw = (3.0 / h**3) * (dw[0]**2.0 + dw[1]**2.0)
        return N, dNdw

    def applyoperator(self, mesh, w):
        '''Apply nonlinear operator N to w to get N(w) in (V^j)'.'''
        Nw = mesh.zeros()
        for p in self._sweepindices(mesh, forward=True):
            Nw[p], _ = self._pointN(mesh.h, w, p)
        return Nw

    def residual(self, mesh, w, ell):
        '''Compute the residual functional for given iterate w.  Note
        ell is a source term in V^j'.'''
        mesh.checklen(w)
        mesh.checklen(ell)
        F = mesh.zeros()
        for p in self._sweepindices(mesh, forward=True):
            F[p], _ = self._pointN(mesh.h, w, p)
            F[p] -= ell[p]
        return F

    def _updatey(self, y, d, phi):
        '''Update y[p] from computed (preliminary) Newton step d.  Ensures
        admissibility and applies Newton-step limitation logic.'''
        d = max(d, phi - y)               # require admissible: y >= phi
        d = min(d, self.newtonupwardmax)
        y += self.args.omega * d          # take step
        return y

    def smoothersweep(self, mesh, y, ell, phi, forward=True):
        '''Do in-place projected nonlinear Gauss-Seidel (PNGS) sweep over the
        interior points p=1,...,m, on the iterate w = g + y.  Here g = mesh.g
        is fixed in the iteration while y varies; the constraint is y >= phi.
        Note that mesh.g must be set before calling this procedure.  Does a
        fixed number of steps of the Newton method (newtonits).'''
        mesh.checklen(y)
        mesh.checklen(ell)
        mesh.checklen(phi)
        self._checkrepairadmissible(mesh, y, phi)
        assert hasattr(mesh, 'g')
        mesh.checklen(mesh.g)
        jaczeros = mesh.zeros()
        for p in self._sweepindices(mesh, forward=forward):
            for k in range(self.args.newtonits):
                N, Jac = self._pointN(mesh.h, mesh.g + y, p)
                if Jac == 0.0:
                    jaczeros[p] = 1.0
                    d = 0.0
                else:
                    d = - (N - ell[p]) / Jac
                if d == 0.0:
                    break
                y[p] = self._updatey(y[p], d, phi[p])
        mesh.WU += self.args.newtonits
        if self.args.showsingular and any(jaczeros != 0.0):
            self.showsingular(jaczeros)

    def source(self, x):
        return 3.0 * (x - 0.5)**2

    def phi(self, x):
        '''Obstacle.'''
        #return np.maximum(0.0, 0.25 * np.sin(3.0 * np.pi * x))
        return 0.25 * np.sin(3.0 * np.pi * x)

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


class PNJacobiPLap(PNGSPLap):
    '''Jacobi smoother derived class of PNGSPLap.'''

    def smoothersweep(self, mesh, y, ell, phi, forward=True):
        '''Do in-place projected nonlinear Jacobi sweep.'''
        mesh.checklen(y)
        mesh.checklen(ell)
        mesh.checklen(phi)
        self._checkrepairadmissible(mesh, y, phi)
        assert hasattr(mesh, 'g')
        mesh.checklen(mesh.g)
        # compute residual and Jacobian at each point
        res, Jac = mesh.zeros(), mesh.zeros()
        for p in self._sweepindices(mesh, forward=True):
            res[p], Jac[p] = self._pointN(mesh.h, mesh.g + y, p)
            res[p] -= ell[p]
        for p in self._sweepindices(mesh, forward=forward):
            for k in range(self.args.newtonits):
                if Jac[p] == 0.0:
                    d = 0.0
                else:
                    d = - res[p] / Jac[p]
                if d == 0.0:
                    break
                y[p] = self._updatey(y[p], d, phi[p])
        mesh.WU += self.args.newtonits
        if self.args.showsingular:
            jaczeros = np.array(Jac == 0.0)
            if any(jaczeros != 0.0):
                self.showsingular(jaczeros)
