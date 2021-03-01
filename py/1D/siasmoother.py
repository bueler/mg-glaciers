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
    smoother for the nonlinear SIA problem
       - (Gamma (s-b)^{n+2} |s'|^{n-1} s')' = m
    with boundary values s(0) = s(L) = 0 on the interval [0,L].  Here b(x) is
    the bed topography, i.e. the obstacle, also with boundary values b(0) =
    b(L) = 0, and m(x) is the surface mass balance.  These data are assumed
    time-independent.  Function s(x) is the solution surface elevation.
    Parameters n and Gamma are set at object construction.'''

    def __init__(self, admissibleeps=1.0e-10, printwarnings=False,
                 g=9.81, rhoi=910.0, nglen=3.0, A=1.0e-16/31556926.0):
        super().__init__(admissibleeps=admissibleeps, printwarnings=printwarnings)
        self.secpera = 31556926.0   # seconds per year
        self.g = g                  # m s-2
        self.rhoi = rhoi            # kg m-3
        self.nglen = nglen
        self.A = A                  # Pa-3 s-1
        self.Gamma = 2.0 * self.A * (self.rhoi * self.g)**self.nglen \
                     / (self.nglen + 2.0)

    def pointresidual(self, mesh, s, ell, p):
        '''Compute the value of the residual linear functional, in V^j', for given
        iterate s(x) in V^j, at one interior hat function psi_p^j:
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
        H = (s[p-1:p+2] - b[p-1:p+2])**(self.nglen + 2.0)     # 3 thicknesses
        ds = s[p:p+2] - s[p-1:p+1]                            # 2 delta s
        np1 = self.nglen + 1.0
        meat = (H[0] + H[1]) * abs(ds[0])**np1 * ds[0] \
               + (H[1] + H[2]) * abs(ds[1])**np1 * ds[1]
        return C * meat - ell[p]

    def smoothersweep(self, mesh, w, ell, phi, forward=True):
        '''Do in-place projected nonlinear Gauss-Seidel sweep over the interior
        points p=1,...,m, for the SIA problem.  Fixed number of steps of the
        Newton method.  FIXME ADD DOC'''
        raise NotImplementedError
