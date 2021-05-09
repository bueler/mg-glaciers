'''Module for SmootherStokes class derived from SmootherObstacleProblem.'''

import numpy as np
from basesmoother import SmootherObstacleProblem

class SmootherStokes(SmootherObstacleProblem):
    '''To evaluate the residual this Jacobi smoother solves the Stokes problem
    for the given geometry by creating an extruded mesh.'''

    def __init__(self, args, admissibleeps=1.0e-10):
        super().__init__(args, admissibleeps=admissibleeps)
        # smoother name
        self.name = 'SmootherStokes'
        # physical parameters
        self.secpera = 31556926.0        # seconds per year
        self.g = 9.81                    # m s-2
        self.rhoi = 910.0                # kg m-3
        self.nglen = 3.0
        self.A3 = 1.0e-16 / self.secpera # Pa-3 s-1;  EISMINT I ice softness
        self.B3 = self.A3**(-1.0/3.0)    # Pa s(1/3);  ice hardness
        self.eps = 0.01
        self.Dtyp = 1.0 / self.secpera   # s-1
        # parameters for initial condition, a Bueler profile; see van der Veen
        #   (2013) section 5.3
        self.buelerL = 10.0e3       # half-width of sheet
        self.buelerH0 = 1000.0      # center thickness
        self.Gamma = 2.0 * self.A3 * (self.rhoi * self.g)**self.nglen
        self.Gamma /= (self.nglen + 2.0)

    def applyoperator(self, mesh, w):
        '''Apply nonlinear operator N to w to get N(w) in (V^j)'.'''
        raise NotImplementedError

    def residual(self, mesh, w, ell):
        '''Compute the residual functional for given iterate w.  Note
        ell is a source term in V^j'.'''
        raise NotImplementedError

    def smoothersweep(self, mesh, y, ell, phi, forward=True):
        '''Do in-place Jacobi smoothing.'''
        raise NotImplementedError

    def jacobisweep(self, mesh, y, ell, phi, forward=True):
        '''Do in-place projected nonlinear Jacobi sweep over the interior
        points p=1,...,m, for the Stokes problem.  On each Newton iteration,
        computes all residuals before updating any iterate values.
        Underrelaxation is expected; try omega = 0.5.'''
        raise NotImplementedError

    def phi(self, x):
        '''For now we have a flat bed.'''
        return np.zeros(np.shape(x))

    def source(self, x):
        '''Continuous source term, i.e. mass balance, for Bueler profile.
        See van der Veen (2013) equations (5.49) and (5.51).  Assumes x
        is a numpy array.'''
        n = self.nglen
        invn = 1.0 / n
        r1 = 2.0 * n + 2.0                   # e.g. 8
        s1 = (1.0 - n) / n                   #     -2/3
        C = self.buelerH0**r1 * self.Gamma   # A_0 in van der Veen is Gamma here
        C /= ( 2.0 * self.buelerL * (1.0 - 1.0 / n) )**n
        xc = self.args.domainlength / 2.0
        X = (x - xc) / self.buelerL          # rescaled coord
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
        '''Default initial shape is the Bueler profile.  See van der Veen (2013)
        equation (5.50).  Assumes x is a numpy array.'''
        n = self.nglen
        p1 = n / (2.0 * n + 2.0)                  # e.g. 3/8
        q1 = 1.0 + 1.0 / n                        #      4/3
        Z = self.buelerH0 / (n - 1.0)**p1         # outer constant
        xc = self.args.domainlength / 2.0
        X = (x - xc) / self.buelerL               # rescaled coord
        Xin = abs(X[abs(X) < 1.0])                # rescaled distance from
                                                  #   center, in ice
        Yin = 1.0 - Xin
        s = np.zeros(np.shape(x))                 # correct outside ice
        s[abs(X) < 1.0] = Z * ( (n + 1.0) * Xin - 1.0 \
                                + n * Yin**q1 - n * Xin**q1 )**p1
        return s
