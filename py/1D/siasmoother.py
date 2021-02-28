'''Module for PNGSSIA class, a derived class of SmootherObstacleProblem,
which is the smoother and coarse-level solver for the Shallow Ice Approximation
(SIA) ice flow model.'''

# TODO:
#   * draft implementation of pointresidual(), smoothersweep() below
#   * test these in test_modules.py
#   * deploy into obstacle.py

from smoother import SmootherObstacleProblem

class PNGSSIA(SmootherObstacleProblem):
    '''Class for the projected nonlinear Gauss-Seidel (PNGS) algorithm as a
    smoother nonlinear SIA problem
       - (Gamma (s-b)^{n+2} |s'|^{n-1} s')' = m
    with boundary values s(0)=0, s(1)=0.  Here b(x) is the bed topography
    and the obstacle and m(x) is the surface mass balance.  (Both are assumed
    time-independent.)  Function s(x) is the solution surface elevation.'''

    def __init__(self, admissibleeps=1.0e-10, printwarnings=False,
                 g=, rhoi=, nglen=, A=):
        super().__init__(admissibleeps=admissibleeps, printwarnings=printwarnings)
        self.g = g
        self.rhoi = rhoi
        self.nglen = nglen
        self.A = A
        self.Gamma = 2.0 * self.A * (self.rhoi * self.g)**self.nglen / (self.nglen + 2.0)  # FIXME CHECK

    def pointresidual(self, mesh, w, ell, p):
        FIXME

    def _diagonalentry(self, mesh, p):
        FIXME essentially the Jacobian entry

    def smoothersweep(self, mesh, w, ell, phi, forward=True):
        '''Do in-place projected nonlinear Gauss-Seidel sweep over the interior
        points p=1,...,m, for the SIA problem
            FIXME SIA in VI form'''
