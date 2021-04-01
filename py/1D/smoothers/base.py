'''Module for abstract base class SmootherObstacleProblem.'''

__all__ = ['SmootherObstacleProblem']

from abc import ABC, abstractmethod
import numpy as np

class SmootherObstacleProblem(ABC):
    '''Abstact base class for a smoother on an obstacle problem.  Works on
    any mesh of class MeshLevel1D.'''

    def __init__(self, args, admissibleeps=1.0e-10):
        self.args = args
        self.admissibleeps = admissibleeps
        self.inadmissible = 0  # count of repaired admissibility violations
        # fix the random seed for repeatability
        np.random.seed(self.args.randomseed)

    def _checkrepairadmissible(self, mesh, w, phi):
        '''Check and repair feasibility.'''
        for p in range(1, mesh.m+1):
            if w[p] < phi[p] - self.admissibleeps:
                if self.args.printwarnings:
                    print('WARNING: repairing inadmissible w[%d]=%e < phi[%d]=%e on level %d (m=%d)' \
                          % (p, w[p], p, phi[p], mesh.j, mesh.m))
                w[p] = phi[p]
                self.inadmissible += 1

    def _sweepindices(self, mesh, forward=True):
        '''Generate indices for sweep.'''
        if forward:
            ind = range(1, mesh.m+1)    # 1,...,m
        else:
            ind = range(mesh.m, 0, -1)  # m,...,1
        return ind

    def showsingular(self, z):
        '''Print a string indicating singular Jacobian points.'''
        Jstr = ''
        for k in range(len(z)):
            Jstr += '-' if z[k] == 0.0 else '*'
        print('%3d singulars: ' % sum(z > 0.0) + Jstr)

    def smoother(self, iters, mesh, w, ell, phi, forward=True, symmetric=False):
        '''Apply iters sweeps of obstacle-problem smoother on mesh to modify w in
        place.'''
        for _ in range(iters):
            self.smoothersweep(mesh, w, ell, phi, forward=forward)
            if symmetric:
                self.smoothersweep(mesh, w, ell, phi, forward=not forward)

    @abstractmethod
    def applyoperator(self, mesh, w):
        '''Apply only the operator to w to generate a linear functional
        in (V^j)'.  Linear: a(w,.).  Nonlinear: N(w)[.].  Generally not
        needed for linear case, so it can raise NotImplementedError.'''

    @abstractmethod
    def residual(self, mesh, w, ell):
        '''Compute the residual functional for given iterate w.  Note
        ell is a source term in V^j'.  Calls _pointresidual() for values.'''

    @abstractmethod
    def smoothersweep(self, mesh, w, ell, phi, forward=True):
        '''Apply one sweep of obstacle-problem smoother on mesh to modify w in
        place.'''

    @abstractmethod
    def phi(self, x):
        '''Evaluate obstacle phi at location(s) x.'''

    @abstractmethod
    def exact_available(self):
        '''Returns True if there is a valid uexact(x) method.'''

    @abstractmethod
    def source(self, x):
        '''Evaluate source function f at location(s) x.'''

    @abstractmethod
    def exact(self, x):
        '''Evaluate exact solution u at location(s) x.  Call exact_available()
        first.  If exact solution is not available this function will raise
        AssertionError or NotImplementedError.'''

    def initial(self, x):
        '''Generate default initial shape.'''
        return np.maximum(self.phi(x), 0.0)
