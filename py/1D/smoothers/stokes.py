'''Module for PNsmootherStokes class derived from SmootherObstacleProblem.'''

import numpy as np
from smoothers.base import SmootherObstacleProblem

class PNsmootherStokes(SmootherObstacleProblem):
    '''To evaluate the residual this Jacobi smoother solves the Stokes problem
    for the given geometry by creating an extruded mesh.'''

    def __init__(self, args, admissibleeps=1.0e-10):
        super().__init__(args, admissibleeps=admissibleeps)
        # smoother name
        self.name = 'PNJac'
        # physical parameters
        self.secpera = 31556926.0        # seconds per year
        self.g = 9.81                    # m s-2
        self.rhoi = 910.0                # kg m-3
        self.nglen = 3.0
        self.A3 = 1.0e-16 / self.secpera # Pa-3 s-1;  EISMINT I ice softness
        self.B3 = self.A3**(-1.0/3.0)    # Pa s(1/3);  ice hardness
        self.eps = 0.01
        self.Dtyp = 1.0 / secpera        # s-1
        # initialize with a pile of ice equal this duration of accumulation
        self.magicinitage = 3000.0 * self.secpera
        # interval [0,xmax]=[0,1800] km will support a centered dome
        self.L = 1800.0e3
        self.xc = 900.0e3                # x coord of center

    def applyoperator(self, mesh, w):
        '''Apply nonlinear operator N to w to get N(w) in (V^j)'.'''
        FIXME NO _pointN()
        mesh.checklen(w)
        Nw = mesh.zeros()
        for p in self._sweepindices(mesh, forward=True):
            Nw[p], _ = self._pointN(mesh, w[p-1:p+2], p)
        return Nw

    def residual(self, mesh, w, ell):
        '''Compute the residual functional for given iterate w.  Note
        ell is a source term in V^j'.'''
        FIXME NO _pointN()
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
                return self.caccum        # upward if accumulation at ice-free
            else:
                return 0.0                # no information on how to move
        else:
            c = - r / delta               # pure Newton step
            c = max(c, phip - yp)         # ensure admissibility: y >= phi
            c = min(c, self.args.siacupmax)  # pre-limit large upward steps
            return c

    def smoothersweep(self, mesh, y, ell, phi, forward=True):
        '''Do in-place Jacobi smoothing.'''
        mesh.checklen(y)
        mesh.checklen(ell)
        mesh.checklen(phi)
        self._checkrepairadmissible(mesh, y, phi)
        assert hasattr(mesh, 'g')
        mesh.checklen(mesh.g)
        if not self.args.jacobi:
            raise NotImplementedError, 'only Jacobi implemented'
        jaczeros = self.jacobisweep(mesh, y, ell, phi, forward=forward)
        if self.args.showsingular and any(jaczeros != 0.0):
            self.showsingular(jaczeros)
        mesh.WU += self.args.newtonits

    def jacobisweep(self, mesh, y, ell, phi, forward=True):
        '''Do in-place projected nonlinear Jacobi sweep over the interior
        points p=1,...,m, for the Stokes problem.  On each Newton iteration,
        computes all residuals before updating any iterate values.
        Underrelaxation is expected; try omega = 0.5.'''
        FIXME

    def phi(self, x):
        '''For now we have a flat bed.'''
        return np.zeros(np.shape(x))

    def exact_available(self):
        return False

    def source(self, x):
        '''Continuous source term, i.e. mass balance, for Bueler profile.
        See van der Veen (2013) equations (5.49) and (5.51).  Assumes x
        is a numpy array.'''
        # parameters for Bueler profile (exact solution) matching Bueler (2016)
        #   see also van der Veen (2013) section 5.3
        buelerL = 750.0e3      # half-width of sheet
        buelerH0 = 3600.0      # center thickness
        n = self.nglen
        invn = 1.0 / n
        r1 = 2.0 * n + 2.0                   # e.g. 8
        s1 = (1.0 - n) / n                   #     -2/3
        C = buelerH0**r1 * self.Gamma        # A_0 in van der Veen is Gamma here
        C /= ( 2.0 * buelerL * (1.0 - 1.0 / n) )**n
        X = (x - self.xc) / buelerL          # rescaled coord
        m = np.zeros(np.shape(x))
        # usual formula for 0 < |X| < 1
        zzz = (abs(X) > 0.0) * (abs(X) < 1.0)
        if any(zzz):
            Xin = abs(X[zzz])
            Yin = 1.0 - Xin
            m[zzz] = (C / buelerL) \
                     * ( Xin**invn + Yin**invn - 1.0 )**(n-1.0) \
                     * ( Xin**s1 - Yin**s1 )
        # fill singular origin with near value
        if any(X == 0.0):
            Xnear = 1.0e-8
            Ynear = 1.0 - Xnear
            m[X == 0.0] = (C / buelerL) \
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
        plt.legend(loc='upper right', fontsize=12.0)
        plt.xticks([], fontsize=12.0)
        plt.ylabel('elevation (m)', fontsize=12.0)
        plt.subplot(2, 1, 2)
        plt.plot(x/1000.0, self.source(x) * self.secpera, 'k')
        plt.xlabel('x (km)', fontsize=12.0)
        plt.ylabel('mass balance (m a-1)', fontsize=12.0)
        plt.xticks([0.0,300.0,600.0,900.0,1200.0,1500.0,1800.0],fontsize=12.0)
        plt.yticks(fontsize=12.0)
        plt.savefig('stokesdatafigure.pdf', bbox_inches='tight')
