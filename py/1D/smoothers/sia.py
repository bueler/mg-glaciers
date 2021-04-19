'''Module for PNsmootherSIA class, a derived class from
SmootherObstacleProblem.  Provides GS and Jacobi smoothers, and an exact
solution, for the shallow ice approximation (SIA) ice flow model.'''

__all__ = ['PNsmootherSIA']

import numpy as np
from smoothers.base import SmootherObstacleProblem

def indentprint(n, s, end='\n'):
    '''Print 2n spaces and then string s.'''
    for _ in range(n):
        print('  ', end='')
    print(s, end=end)

class PNsmootherSIA(SmootherObstacleProblem):
    '''Class for the projected nonlinear Gauss-Seidel (PNGS) and Jacobi
    (PNJacobi) algorithms as smoothers for the nonlinear SIA obstacle problem
    for y in K = {v >= phi}:
       N(g + y)[v-y] >= ell[v-y]
    for all v in K, where
       N(w)[v] = int_0^L Gamma (w - b + eps)^{p+1} |w'|^{p-2} w' v' dx
    with boundary values w(0) = w(L) = 0 on the interval [0,L].  Here b(x) is
    the bed topography and ell[v] = <a,v> on the finest level, where a(x)
    is the surface mass balance.  The data a(x), b(x) are assumed
    time-independent.  Function w = g + y is the solution surface elevation.
    Parameters n and Gamma are set at object construction.  Note p = n+1
    in the p-Laplacian interpretation.

    The smoother uses a fixed number of Newton iterations (-newtonits) at
    each point.  The Newton iteration uses other "protection" parameters.
    The GS smoother applies an Armijo criterion for its 1D line search.

    This class can also compute the "Bueler profile" exact solution.'''

    def __init__(self, args, admissibleeps=1.0e-10):
        super().__init__(args, admissibleeps=admissibleeps)
        # smoother name
        self.name = 'PNJac' if self.args.jacobi else 'PNGS'
        # physical parameters
        self.secpera = 31556926.0        # seconds per year
        self.g = 9.81                    # m s-2
        self.rhoi = 910.0                # kg m-3
        self.nglen = 3.0
        self.A = 1.0e-16 / self.secpera  # Pa-3 s-1
        # convenience constant
        self.Gamma = 2.0 * self.A * (self.rhoi * self.g)**self.nglen \
                     / (self.nglen + 2.0)
        # important powers
        self.pp = self.nglen + 1.0                      # = 4; p-Laplacian
        self.rr = (2.0 * self.nglen + 2.0) / self.nglen # = 8/3; error reporting
        # parameters used in Newton solver and pointupdate()
        self.ctol = 1.0e-3          # 1 mm; no Armijo if c is this small
        self.caccum = 10.0          # m
        self.cupmax = 500.0         # m; never move surface up by more than this
                                    #    perhaps should be comparable to 0.2 or
                                    #    0.1 of max ice thickness
        # initialize with a pile of ice equal this duration of accumulation
        self.magicinitage = 3000.0 * self.secpera
        # interval [0,xmax]=[0,1800] km
        self.L = 1800.0e3
        self.xc = 900.0e3           # x coord of center
        # parameters for Bueler profile (exact solution) matching Bueler (2016);
        #   see also van der Veen (2013) section 5.3
        self.buelerL = 750.0e3      # half-width of sheet
        self.buelerH0 = 3600.0      # center thickness

    def _pointN(self, mesh, wpatch, p):
        '''Compute nonlinear operator value N(w)[psi_p^j], for
        given iterate w(x) in V^j, at one hat function psi_p^j:
           N(w)[psi_p^j] = int_I Gamma (w(x) - b(x) + eps)^{p+1}
                                       * |w'(x)|^{p-2} w'(x) (psi_p^j)'(x) dx
        where I = [x_p - h, x_p + h].  Approximates using the trapezoid rule.
        Also return dNdw, the derivative of N(w)[psi_p^j] with respect to w[p].
        Note that wpatch = [w[p-1], w[p], w[p+1]].'''
        assert hasattr(mesh, 'h')
        assert hasattr(mesh, 'b')
        assert len(wpatch) == 3
        eps = self.args.siaeps
        tau = (wpatch - mesh.b[p-1:p+2] + eps)**(self.pp + 1.0)
        dtau = (self.pp + 1.0) * (wpatch[1] - mesh.b[p] + eps)**self.pp
        ds = wpatch[1:] - wpatch[:2]
        mu = abs(ds)**(self.pp - 2.0) * ds
        dmu = (self.pp - 1.0) * abs(ds)**(self.pp - 2.0)
        C = self.Gamma / (2.0 * mesh.h**(self.pp-1.0))
        N = C * ( (tau[0] + tau[1]) * mu[0] - (tau[1] + tau[2]) * mu[1] )
        dNdw = C * ( dtau * (mu[0] - mu[1]) + \
                     (tau[0] + tau[1]) * dmu[0] + (tau[1] + tau[2]) * dmu[1] )
        return N, dNdw

    def applyoperator(self, mesh, w):
        '''Apply nonlinear operator N to w to get N(w) in (V^j)'.'''
        mesh.checklen(w)
        Nw = mesh.zeros()
        for p in self._sweepindices(mesh, forward=True):
            Nw[p], _ = self._pointN(mesh, w[p-1:p+2], p)
        return Nw

    def residual(self, mesh, w, ell):
        '''Compute the residual functional for given iterate w.  Note
        ell is a source term in V^j'.'''
        mesh.checklen(w)
        mesh.checklen(ell)
        F = mesh.zeros()
        for p in self._sweepindices(mesh, forward=True):
            F[p], _ = self._pointN(mesh, w[p-1:p+2], p)
            F[p] -= ell[p]
        return F

    def _pointupdate(self, r, delta, yp, phip, ellp):
        '''Compute candidate update value c, for y[p] += c, from pointwise
        residual r = rho(0) and pointwise Jacobian delta = rho'(0).'''
        if delta == 0.0:
            if ellp > 0.0 and yp == 0.0:
                return self.caccum    # move upward if accumulation at ice-free
            else:
                return 0.0            # no information on how to move
        else:
            c = - r / delta           # pure Newton step
            c = max(c, phip - yp)     # ensure admissibility: y >= phi
            c = min(c, self.cupmax)   # pre-limit large upward steps
            return c

    def smoothersweep(self, mesh, y, ell, phi, forward=True):
        '''Do either in-place GS or Jacobi smoothing.'''
        mesh.checklen(y)
        mesh.checklen(ell)
        mesh.checklen(phi)
        self._checkrepairadmissible(mesh, y, phi)
        assert hasattr(mesh, 'g')
        mesh.checklen(mesh.g)
        if self.args.jacobi:
            jaczeros = self.jacobisweep(mesh, y, ell, phi, forward=forward)
        else:
            jaczeros = self.gssweep(mesh, y, ell, phi, forward=forward)
        if self.args.showsingular and any(jaczeros != 0.0):
            self.showsingular(jaczeros)
        mesh.WU += self.args.newtonits

    def gssweep(self, mesh, y, ell, phi, forward=True):
        '''Do in-place projected nonlinear Gauss-Seidel (PNGS) sweep over the
        interior points p=1,...,m, for the SIA problem on the iterate w = g + y.
        Here g = mesh.g is fixed in the iteration and y varies, and the
        constraint is y >= phi.  Note that mesh.g and mesh.b (bed elevation)
        must be set before calling this procedure.  Does a fixed number of
        steps of the Newton method (newtonits).'''
        jaczeros = mesh.zeros()
        arm_alpha = 1.0e-4
        arm_rho = 2.0
        armijo = 0  # count of added Armijo point residual evaluations
        # update each y[p] value
        for p in self._sweepindices(mesh, forward=forward):
            for _ in range(self.args.newtonits):
                wpatch = mesh.g[p-1:p+2] + y[p-1:p+2]
                r, delta = self._pointN(mesh, wpatch, p)
                r -= ell[p]               # actual residual
                c = self._pointupdate(r, delta, y[p], phi[p], ell[p])
                if delta == 0.0:
                    jaczeros[p] = 1.0
                # if c is reasonable, apply Armijo criterion to reduce c until
                #     sufficient decrease (Kelley section 1.6)
                if delta != 0.0 and abs(c) > self.ctol:
                    reduce = 1.0
                    for _ in range(20):
                        zpatch = wpatch.copy()
                        zpatch[1] += self.args.omega * c / reduce
                        rnew, _ = self._pointN(mesh, zpatch, p)
                        rnew -= ell[p]
                        if abs(rnew) < (1.0 - arm_alpha / reduce) * abs(r):
                            c /= reduce
                            break
                        reduce *= arm_rho
                        armijo += 1
                y[p] += self.args.omega * c
        if self.args.armijomonitor:
            arm_av = armijo / (self.args.newtonits * mesh.m)
            indentprint(self.args.J - mesh.j,
                        '  gssweep() on level %d: added armijo evals = %d (fraction %.2f)' % (mesh.j, armijo, arm_av))
        return jaczeros

    def jacobisweep(self, mesh, y, ell, phi, forward=True):
        '''Do in-place projected nonlinear Jacobi sweep over the interior
        points p=1,...,m, for the SIA problem.  On each Newton iteration,
        computes all residuals before updating any iterate values.
        Compare gssweep(); note we cannot do Armijo line search because we
        want to only evaluate residuals twice.  Underrelaxation is expected;
        try omega = 0.5.'''
        jaczeros = mesh.zeros()
        r, delta = mesh.zeros(), mesh.zeros()
        for _ in range(self.args.newtonits):
            # compute residuals and Jacobians at each point
            for p in self._sweepindices(mesh, forward=True):
                wpatch = mesh.g[p-1:p+2] + y[p-1:p+2]
                r[p], delta[p] = self._pointN(mesh, wpatch, p)
                r[p] -= ell[p]
            # update each y[p] value
            for p in self._sweepindices(mesh, forward=forward):
                if delta[p] == 0.0:
                    jaczeros[p] = 1.0
                c = self._pointupdate(r[p], delta[p], y[p], phi[p], ell[p])
                y[p] += self.args.omega * c
        return jaczeros

    def phi(self, x):
        '''For now we have a flat bed.'''
        if self.args.siacase == 'profile':
            return np.zeros(np.shape(x))
        elif self.args.siacase == 'bumpy':
            return (1200.0 - 1000.0 * x / self.L) \
                   * np.sin(8.0 * np.pi * x / self.L)

    def exact_available(self):
        '''For now we use the Bueler profile, an exact solution.'''
        return (self.args.siacase == 'profile')

    def exact(self, x):
        '''Exact solution (Bueler profile).  See van der Veen (2013) equation
        (5.50).  Assumes x is a numpy array.'''
        assert self.exact_available()
        n = self.nglen
        p1 = n / (2.0 * n + 2.0)                  # e.g. 3/8
        q1 = 1.0 + 1.0 / n                        #      4/3
        Z = self.buelerH0 / (n - 1.0)**p1         # outer constant
        X = (x - self.xc) / self.buelerL          # rescaled coord
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
        X = (x - self.xc) / self.buelerL          # rescaled coord
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
        return self.phi(x) + self.magicinitage * np.maximum(0.0,self.source(x))

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
