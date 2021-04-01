'''Module for PNGSPLap and PNJacobiPLap classes, a derived class from
SmootherObstacleProblem.  Provides smoothers and exact solutions
for the p-Laplacian obstacle problem.'''

__all__ = ['PNGSPLap', 'PNJacobiPLap']

import numpy as np
from smoothers.base import SmootherObstacleProblem

class PNGSPLap(SmootherObstacleProblem):
    '''Smoother for the p-Laplacian obstacle problem.  Compare PGSPoisson
    and PNGSSIA.  (Jacobi version currently not implemented.)'''

    def __init__(self, args, admissibleeps=1.0e-10):
        super().__init__(args, admissibleeps=admissibleeps)
        self.args = args
        # Newton's method parameters (used in PNGS smoother)
        self.newtonits = 2          # number of Newton its
        self.newtondtol = 1.0e-8    # don't continue Newton if d is this small

    def residual(self, mesh, w, ell):
        pass

    def smoothersweep(self, mesh, w, ell, phi, forward=True):
        pass

    def source(self, x):
        pass

    def phi(self, x):
        '''Obstacle.'''
        return np.maximum(0.0, np.sin(3.0 * np.pi * x))

    def exact_available(self):
        return True

    def _solveforb(self):
        '''Find constant in exact solution by root finding on system
            u(b) = phi(b)
            u'(b) = phi'(b)
        using
            u(x) = 0.5 * x (1-x) + C
            phi(x) = sin(3 pi x)
        knowing that b is in (5/6,1) and close to 0.837.'''
        def ff(b):
            return 0.5 - b - 3.0 * np.pi * np.cos(3.0 * np.pi * b)
        def dff(b):
            return 9.0 * np.pi**2 * np.sin(3.0 * np.pi * b)
        b = 0.837
        for k in range(7):
            b = b - ff(b) / dff(b)
        return b

    def exact(self, x):
        '''Exact solution.'''
        b = self._solveforb()
        C = np.sin(3.0 * np.pi * b) - 0.5 * b * (1.0 - b)
        u = self.phi(x)
        mid = (x > 1.0-b) * (x < b)
        u[mid] = 0.5 * x[mid] * (1.0 - x[mid]) + C
        return u

    def datafigure(self, mesh):
        '''Show exact solution in a basic figure.'''
        x = mesh.xx()
        import matplotlib.pyplot as plt
        plt.plot(x, self.phi(x), 'k', label='obstacle')
        plt.plot(x, self.exact(x), 'k--', label='$u_{exact}$')
        plt.legend(fontsize=12.0)
        plt.xlabel('x', fontsize=12.0)
        plt.savefig('plapfigure.pdf', bbox_inches='tight')
