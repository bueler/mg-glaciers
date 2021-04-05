'''Module for PNGSSIA and PNJacobiSIA classes, a derived class from
SmootherObstacleProblem.  Provides smoothers and exact solutions
for the shallow ice approximation (SIA) ice flow model.'''

__all__ = ['PNGSSIA', 'PNJacobiSIA']

import numpy as np
from smoothers.base import SmootherObstacleProblem

class PNGSSIA(SmootherObstacleProblem):
    '''Class for the projected nonlinear Gauss-Seidel (PNGS) algorithm as a
    smoother for the nonlinear SIA obstacle problem for y in K = {v >= phi}:
       N(g + y)[v-y] >= ell[v-y]
    for all v in K, where
       N(w)[v] = int_0^L Gamma (w - b + eps)^{p+1} |w'|^{p-2} w' v' dx
    with boundary values w(0) = w(L) = 0 on the interval [0,L].  Here b(x) is
    the bed topography and ell[v] = <a,v> on the finest level, where a(x)
    is the surface mass balance.  The data a(x), b(x) are assumed
    time-independent.  Function w = g + y is the solution surface elevation.
    Parameters n and Gamma are set at object construction.  Note p = n+1
    in the p-Laplacian interpretation.

    The PNGS smoother uses a fixed number of Newton iterations (-newtonits) at
    each point.  The Newton iteration uses two other parameters:
      * newtonupwardmax:  when mass balance is positive at an ice-free location
        then the Newton step is formally infinite, so we limit the step.

    This class can also compute the "Bueler profile" exact solution.'''

    def __init__(self, args, admissibleeps=1.0e-10,
                 g=9.81, rhoi=910.0, nglen=3.0, A=1.0e-16/31556926.0):
        super().__init__(args, admissibleeps=admissibleeps)
        self.args = args
        self.secpera = 31556926.0   # seconds per year
        self.g = g                  # m s-2
        self.rhoi = rhoi            # kg m-3
        self.nglen = nglen
        self.A = A                  # Pa-3 s-1
        # convenience constant
        self.Gamma = 2.0 * self.A * (self.rhoi * self.g)**self.nglen \
                     / (self.nglen + 2.0)
        # important powers
        self.pp = self.nglen + 1.0                      # = 4; p-Laplacian
        self.rr = (2.0 * self.nglen + 2.0) / self.nglen # = 8/3; error reporting
        # step surface upward this much if Jacobian is zero and the
        #   source term corresponds to accumulation at the point
        self.daccumulation = 10.0   # m
        # parameter in PNGS and PNJacobi smoothers
        self.newtonupwardmax = 5000.0 # never move surface up by more than this
        # initialize with a pile of ice equal this duration of accumulation
        self.magicinitage = 3000.0 * self.secpera
        # parameters for Bueler profile (exact solution) matching Bueler (2016);
        #   see also van der Veen (2013) section 5.3
        self.buelerL = 750.0e3      # half-width of sheet
        self.buelerH0 = 3600.0      # center thickness
        self.buelerxc = 900.0e3     # x coord of center in [0,xmax]=[0,1800] km

    def _pointN(self, h, b, w, p):
        '''Compute nonlinear operator value N(w)[psi_p^j], for
        given iterate w(x) in V^j, at one hat function psi_p^j:
           N(w)[psi_p^j] = int_I Gamma (w(x) - b(x) + eps)^{p+1}
                                       * |w'(x)|^{p-2} w'(x) (psi_p^j)'(x) dx
        where I = [x_p - h, x_p + h].  Approximates using the trapezoid rule.
        Also return dNdw, the derivative of N(w)[psi_p^j] with respect to w[p].'''
        eps = self.args.siaeps
        tau = (w[p-1:p+2] - b[p-1:p+2] + eps)**(self.pp + 1.0)
        dtau = (self.pp + 1.0) * (w[p] - b[p] + eps)**self.pp
        ds = w[p:p+2] - w[p-1:p+1]
        mu = abs(ds)**(self.pp - 2.0) * ds
        dmu = (self.pp - 1.0) * abs(ds)**(self.pp - 2.0)
        C = self.Gamma / (2.0 * h**(self.pp-1.0))
        N = C * ( (tau[0] + tau[1]) * mu[0] - (tau[1] + tau[2]) * mu[1] )
        dNdw = C * ( dtau * (mu[0] - mu[1]) + \
                     (tau[0] + tau[1]) * dmu[0] + (tau[1] + tau[2]) * dmu[1] )
        return N, dNdw

    def applyoperator(self, mesh, w):
        '''Apply nonlinear operator N to w to get N(w) in (V^j)'.'''
        Nw = mesh.zeros()
        for p in self._sweepindices(mesh, forward=True):
            Nw[p], _ = self._pointN(mesh.h, mesh.b, w, p)
        return Nw

    def residual(self, mesh, w, ell):
        '''Compute the residual functional for given iterate w.  Note
        ell is a source term in V^j'.'''
        mesh.checklen(w)
        mesh.checklen(ell)
        assert hasattr(mesh, 'b')
        F = mesh.zeros()
        for p in self._sweepindices(mesh, forward=True):
            F[p], _ = self._pointN(mesh.h, mesh.b, w, p)
            F[p] -= ell[p]
        return F

    def _updatey(self, y, d, phi):
        '''Update y[p] from computed (preliminary) Newton step d.  Ensures
        admissibility and applies Newton-step limitation logic.'''
        d = max(d, phi - y)                 # require admissible: y >= phi
        d = min(d, self.newtonupwardmax)    # limit huge upward steps
        y += self.args.omega * d            # take step
        return y

    def smoothersweep(self, mesh, y, ell, phi, forward=True):
        '''Do in-place projected nonlinear Gauss-Seidel (PNGS) sweep over the
        interior points p=1,...,m, for the SIA problem on the iterate w = g + y.
        Here g = mesh.g is fixed in the iteration and y varies, and the
        constraint is y >= phi.  Note that mesh.g and mesh.b (bed elevation)
        must be set before calling this procedure.  Does a fixed number of
        steps of the Newton method (newtonits).'''
        mesh.checklen(y)
        mesh.checklen(ell)
        mesh.checklen(phi)
        self._checkrepairadmissible(mesh, y, phi)
        assert hasattr(mesh, 'b')
        mesh.checklen(mesh.b)
        assert hasattr(mesh, 'g')
        mesh.checklen(mesh.g)
        jaczeros = mesh.zeros()
        for p in self._sweepindices(mesh, forward=forward):
            for k in range(self.args.newtonits):
                N, Jac = self._pointN(mesh.h, mesh.b, mesh.g + y, p)
                if Jac == 0.0:
                    jaczeros[p] = 1.0
                    d = self.daccumulation if ell[p] > 0.0 else 0.0
                else:
                    d = - (N - ell[p]) / Jac
                if d == 0.0:
                    break
                y[p] = self._updatey(y[p], d, phi[p])
        mesh.WU += self.args.newtonits
        if self.args.showsingular and any(jaczeros != 0.0):
            self.showsingular(jaczeros)

    def phi(self, x):
        '''For now we have a flat bed.'''
        return np.zeros(np.shape(x))

    def exact_available(self):
        '''For now we use the Bueler profile, an exact solution.'''
        return True

    def exact(self, x):
        '''Exact solution (Bueler profile).  See van der Veen (2013) equation
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

    def initial(self, x):
        '''Default initial shape is a stack of ice where surface mass
        balance is positive.'''
        return self.magicinitage * np.maximum(0.0,self.source(x))

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

    def smoothersweep(self, mesh, y, ell, phi, forward=True):
        '''Do in-place projected nonlinear Jacobi sweep over the interior
        points p=1,...,m, for the SIA problem.  Compare PNGSSIA.smoothersweep().
        Underrelaxation is expected; try omega = 0.5.'''
        mesh.checklen(y)
        mesh.checklen(ell)
        mesh.checklen(phi)
        self._checkrepairadmissible(mesh, y, phi)
        assert hasattr(mesh, 'b')
        mesh.checklen(mesh.b)
        assert hasattr(mesh, 'g')
        mesh.checklen(mesh.g)
        # compute residual and Jacobian at each point
        res, Jac = mesh.zeros(), mesh.zeros()
        for p in self._sweepindices(mesh, forward=True):
            res[p], Jac[p] = self._pointN(mesh.h, mesh.b, mesh.g + y, p)
            res[p] -= ell[p]
        # update each w[p] value
        for p in self._sweepindices(mesh, forward=forward):
            for k in range(self.args.newtonits):
                if Jac[p] == 0.0:
                    d = self.daccumulation if ell[p] > 0.0 else 0.0
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