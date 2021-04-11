'''Module for PJacobiStokes class, a derived class of SmootherObstacleProblem,
which is the smoother and coarse-level solver for the Stokes ice flow model.'''

# THOUGHTS:
#   * use Firedrake to evaluate the residual
#   * use projected weighted Jacobi because pointwise residual modifications
#     are not obvious and might be expensive

import numpy as np
from smoothers.base import SmootherObstacleProblem

class PJacobiStokes(SmootherObstacleProblem):
    '''FIXME'''

    def __init__(self, admissibleeps=1.0e-10, printwarnings=False,
                 g=, rhoi=, nglen=, A=):
        super().__init__(admissibleeps=admissibleeps, printwarnings=printwarnings)
        self.g = g
        self.rhoi = rhoi
        self.nglen = nglen
        self.A = A
        self.name = 'PNJac'

    def smoothersweep(self, mesh, w, ell, phi, forward=True):
        '''Do in-place projected nonlinear Jacobi sweep over the interior
        points p=1,...,m, for the Stokes problem, calling Firedrake to solve.'''
        FIXME
