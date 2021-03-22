'''Module for PNGSSIA class, a derived class of SmootherObstacleProblem,
which is the smoother and coarse-level solver for the Shallow Ice Approximation
(SIA) ice flow model.'''

__all__ = ['PNGSSIA', 'PNJacobiSIA']

import numpy as np
from smoother import SmootherObstacleProblem

class PNGSSIA(SmootherObstacleProblem):
    '''Class for the projected nonlinear Gauss-Seidel (PNGS) algorithm as a
    smoother for the nonlinear SIA problem
       - (Gamma (s-b)^{n+2} |s'|^{n-1} s')' = m
    with boundary values s(0) = s(L) = 0 on the interval [0,L].  Here b(x) is
    the bed topography, i.e. the obstacle, also with boundary values b(0) =
    b(L) = 0, and m(x) is the surface mass balance.  These data are assumed
    time-independent.  Function s(x) is the solution surface elevation.
    Parameters n and Gamma are set at object construction.

    The PNGS smoother sweep uses a fixed number of Newton iterations at each
    point.  The Newton step involves three parameters associated to the
    degeneracy of the diffusion:
      * Hmin, slopemin:  protect against division by zero in computing
                         Newton step
      * newtondmax:      when mass balance is positive at an ice-free
                         location then the Newton step is formally
                         infinite, so we limit the step.'''

    def __init__(self, args, admissibleeps=1.0e-10,
                 g=9.81, rhoi=910.0, nglen=3.0, A=1.0e-16/31556926.0):
        super().__init__(args, admissibleeps=admissibleeps)
        self.secpera = 31556926.0   # seconds per year
        self.g = g                  # m s-2
        self.rhoi = rhoi            # kg m-3
        self.nglen = nglen
        self.A = A                  # Pa-3 s-1
        self.Gamma = 2.0 * self.A * (self.rhoi * self.g)**self.nglen \
                     / (self.nglen + 2.0)
        # parameters for Bueler profile (exact solution) matching Bueler (2016);
        #   see also van der Veen (2013) section 5.3
        self.buelerL = 750.0e3      # half-width of sheet
        self.buelerH0 = 3600.0      # center thickness
        self.buelerxc = 900.0e3     # x coord of center in [0,xmax] = [0,1800] km
        # numerical parameters in smoother
        self.newtonits = 2          # number of Newton its
        self.newtondtol = 1.0e-8    # don't continue Newton if d is this small
        self.newtondmax = 1000.0    # never move surface by more than this amount
        self.newtonHmin = 100.0     # protect divide by zero in smoother
        self.newtonslopemin = 1.0e-4 # protect divide by zero in smoother

    def pointresidual(self, mesh, s, ell, p):
        '''Compute the value of the residual linear functional, in V^j', for
        given iterate s(x) in V^j, at one interior hat function psi_p^j:
           F(s)[psi_p^j] = int_0^L Gamma (s(x) - phi(x))^{n+2} |s'(x)|^{n-1} s'(x) dx
                           - ell(psi_p^j)
        Input ell is in V^j'.  Input mesh is of class MeshLevel1D, with attached
        obstacle b = mesh.phi.'''
        mesh.checklen(s)
        mesh.checklen(ell)
        assert hasattr(mesh, 'phi')
        b = mesh.phi
        mesh.checklen(b)
        assert 1 <= p <= mesh.m
        C = self.Gamma / (2.0 * mesh.h**self.nglen)
        HP = (s[p-1:p+2] - b[p-1:p+2])**(self.nglen + 2.0)  # 3 thickness powers
        ds = s[p:p+2] - s[p-1:p+1]                          # 2 delta s
        mu = abs(ds)**(self.nglen - 1.0) * ds
        meat = (HP[0] + HP[1]) * mu[0] - (HP[1] + HP[2]) * mu[1]
        return C * meat - ell[p]

    def _pointjacobian(self, mesh, s, p):
        '''The derivative of pointresidual() with respect to s[p].'''
        mesh.checklen(s)
        assert hasattr(mesh, 'phi')
        b = mesh.phi
        mesh.checklen(b)
        assert 1 <= p <= mesh.m
        C = self.Gamma / (2.0 * mesh.h**self.nglen)
        # protects divide by zero in smoother:
        Hpos = np.maximum(s[p-1:p+2] - b[p-1:p+2], self.newtonHmin)
        HP = Hpos**(self.nglen + 2.0)
        # protects divide by zero in smoother:
        ads = np.maximum(abs(s[p:p+2] - s[p-1:p+1]), mesh.h * self.newtonslopemin)
        dmu = self.nglen * ads**(self.nglen - 1.0)
        return C * ( (HP[0] + HP[1]) * dmu[0] + (HP[1] + HP[2]) * dmu[1] )

    def smoothersweep(self, mesh, s, ell, phi, forward=True):
        '''Do in-place projected nonlinear Gauss-Seidel sweep over the interior
        points p=1,...,m, for the SIA problem.  Fixed number of steps of the
        Newton method (newtonits).  Protection against taking huge steps
        (newtondmax).  Avoids further iteration if first step is small
        (newtondtol).'''
        self._checkrepairadmissible(mesh, s, phi)
        for p in self._sweepindices(mesh, forward=forward):
            for k in range(self.newtonits):
                d = - self.pointresidual(mesh, s, ell, p) / self._pointjacobian(mesh, s, p)
                d = max(d, phi[p] - s[p])                     # require admissible
                d = np.sign(d) * min(abs(d), self.newtondmax) # limit huge steps
                s[p] += self.args.omega * d                   # take step
                if abs(self.args.omega * d) < self.newtondtol: # tiny step
                    break
        mesh.WU += self.newtonits  # overcount WU if many points are active

    def phi(self, x):
        '''Generally the bed elevations depend on self.args, but for now we
        have a flat bed.'''
        return np.zeros(np.shape(x))

    def exact_available(self):
        '''Generally whether there is an exact solution depends on self.args
        but for now we just do the Bueler profile.'''
        return True

    def exact(self, x):
        '''Exact solution Bueler profile.  See van der Veen (2013) equation
        (5.50).  Assumes x is a numpy array.'''
        assert self.exact_available()
        n = self.nglen
        p1 = n / (2.0 * n + 2.0)                  # e.g. 3/8
        q1 = 1.0 + 1.0 / n                        #      4/3
        Z = self.buelerH0 / (n - 1.0)**p1         # outer constant
        X = (x - self.buelerxc) / self.buelerL    # rescaled coord
        Xin = abs(X[abs(X) < 1.0])                # rescaled distance from
                                                  #   center, in ice
        Yin = 1.0 - Xin
        s = np.zeros(np.shape(x))                 # correct outside ice
        s[abs(X) < 1.0] = Z * ( (n + 1.0) * Xin - 1.0 \
                                + n * Yin**q1 - n * Xin**q1 )**p1
        return s

    def source(self, x):
        '''Continuous source term, i.e. mass balance, for Bueler profile.
        See van der Veen (2013) equations (5.49) and (5.51).  Assumes x
        is a numpy array.'''
        n = self.nglen
        invn = 1.0 / n
        r1 = 2.0 * n + 2.0                        # e.g. 8
        s1 = (1.0 - n) / n                        #     -2/3
        C = self.buelerH0**r1 * self.Gamma        # A_0 in van der Veen is Gamma here
        C /= ( 2.0 * self.buelerL * (1.0 - 1.0 / n) )**n
        X = (x - self.buelerxc) / self.buelerL    # rescaled coord
        m = np.zeros(np.shape(x))
        # usual formula for 0 < |X| < 1
        zzz = (abs(X) > 0.0) * (abs(X) < 1.0)
        if any(zzz):
            Xin = abs(X[zzz])
            Yin = 1.0 - Xin
            m[zzz] = (C / self.buelerL) \
                     * ( Xin**invn + Yin**invn - 1.0 )**(n-1.0) \
                     * ( Xin**s1 - Yin**s1 )
        # fill singular origin with near value
        if any(X == 0.0):
            Xnear = 1.0e-8
            Ynear = 1.0 - Xnear
            m[X == 0.0] = (C / self.buelerL) \
                          * ( Xnear**invn + Ynear**invn - 1.0 )**(n-1.0) \
                          * ( Xnear**s1 - Ynear**s1 )
        # extend by ablation
        if any(abs(X) >= 1.0):
            m[abs(X) >= 1.0] = min(m)
        return m

    def datafigure(self, mesh):
        '''Show data phi, source, exact in a basic figure.'''
        x = mesh.xx()
        import matplotlib.pyplot as plt
        plt.subplot(2, 1, 1)
        plt.plot(x/1000.0, self.phi(x), 'k', label='bed')
        plt.plot(x/1000.0, self.exact(x), 'k--', label='ice surface')
        plt.legend(loc='upper right', fontsize=12.0)
        plt.xticks([], fontsize=12.0)
        plt.ylabel('elevation (m)', fontsize=12.0)
        plt.subplot(2, 1, 2)
        plt.plot(x/1000.0, self.source(x) * self.secpera, 'k')
        plt.xlabel('x (km)', fontsize=12.0)
        plt.ylabel('mass balance (m a-1)', fontsize=12.0)
        plt.xticks([0.0,300.0,600.0,900.0,1200.0,1500.0,1800.0],fontsize=12.0)
        plt.yticks(fontsize=12.0)
        plt.savefig('siadatafigure.pdf', bbox_inches='tight')


class PNJacobiSIA(PNGSSIA):

    def _jacobian(self, mesh, s):
        '''Compute the Jacobian at each point, returning a vector.  Calls
        _pointjacobian() for values.'''
        mesh.checklen(s)
        J = mesh.zeros()
        for p in range(1, mesh.m+1):
            J[p] = self._pointjacobian(mesh, s, p)
        return J

    def smoothersweep(self, mesh, s, ell, phi, forward=True):
        '''Do in-place projected nonlinear Jacobi sweep over the interior
        points p=1,...,m, for the SIA problem.  Compare PNGSSIA.smoothersweep()
        and PJacobiPoisson.smoothersweep().  Underrelaxation is expected;
        try omega = 0.8.'''
        self._checkrepairadmissible(mesh, s, phi)
        res = self.residual(mesh, s, ell)
        Jac = self._jacobian(mesh, s)
        for p in self._sweepindices(mesh, forward=forward):
            for k in range(self.newtonits):
                d = - res[p] / Jac[p]
                d = max(d, phi[p] - s[p])                     # require admissible
                d = np.sign(d) * min(abs(d), self.newtondmax) # limit huge steps
                s[p] += self.args.omega * d                   # take step
                if abs(self.args.omega * d) < self.newtondtol: # tiny step
                    break
        mesh.WU += self.newtonits  # overcount WU if many points are active
