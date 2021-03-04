'''Module for SmootherObstacleProblem class and its derived class PGSPoisson.'''

__all__ = ['SmootherObstacleProblem', 'PGSPoisson', 'PJacobiPoisson']

from abc import ABC, abstractmethod
import numpy as np

class SmootherObstacleProblem(ABC):
    '''Abstact base class for a smoother on an obstacle problem.  Works on
    any mesh of class MeshLevel1D.'''

    def __init__(self, args, admissibleeps=1.0e-10):
        self.args = args
        self.admissibleeps = admissibleeps
        # fix the random seed for repeatability
        np.random.seed(self.args.randomseed)

    def _checkrepairadmissible(self, mesh, w, phi):
        '''Check and repair feasibility.'''
        infeascount = 0
        for p in range(1, mesh.m+1):
            if w[p] < phi[p] - self.admissibleeps:
                if self.args.printwarnings:
                    print('WARNING: repairing nonfeasible w[%d]=%e < phi[%d]=%e on level %d (m=%d)' \
                          % (p, w[p], p, phi[p], mesh.j, mesh.m))
                w[p] = phi[p]
                infeascount += 1
        return infeascount

    def _sweepindices(self, mesh, forward=True):
        '''Generate indices for sweep.'''
        if forward:
            ind = range(1, mesh.m+1)    # 1,...,m
        else:
            ind = range(mesh.m, 0, -1)  # m,...,1
        return ind

    @abstractmethod
    def pointresidual(self, mesh, w, ell, p):
        '''Compute the residual functional for given iterate w at a point p.'''

    def residual(self, mesh, w, ell):
        '''Compute the residual functional for given iterate w.  Note
        ell is a source term in V^j'.  Calls _pointresidual() for values.'''
        mesh.checklen(w)
        mesh.checklen(ell)
        F = mesh.zeros()
        for p in range(1, mesh.m+1):
            F[p] = self.pointresidual(mesh, w, ell, p)
        return F

    @abstractmethod
    def smoothersweep(self, mesh, w, ell, phi, forward=True):
        '''Apply obstacle-problem smoother on mesh to modify w in place.'''

    @abstractmethod
    def phi(self, x):
        '''Evaluate obstacle at location(s) x.'''

    @abstractmethod
    def exact_available(self):
        '''Returns True if there is a valid uexact(x) method.'''

def _poissondiagonalentry(mesh, p):
    '''Compute the diagonal value of a(.,.) at hat function psi_p^j:
       a(psi_p,psi_p) = int_0^1 (psi_p^j)'(x)^2 dx
    Input mesh is of class MeshLevel1D.'''
    assert 1 <= p <= mesh.m
    return 2.0 / mesh.h

class PGSPoisson(SmootherObstacleProblem):
    '''Class for the projected Gauss-Seidel (PGS) algorithm as a smoother
    for the linear Poisson equation -u''=f with u(0)=u(1)=0.'''

    def pointresidual(self, mesh, w, ell, p):
        '''Compute the value of the residual linear functional, in V^j', for given
        iterate w, at one interior hat function psi_p^j:
           F(w)[psi_p^j] = int_0^1 w'(x) (psi_p^j)'(x) dx - ell(psi_p^j)
        Input ell is in V^j'.  Input mesh is of class MeshLevel1D.'''
        mesh.checklen(w)
        mesh.checklen(ell)
        assert 1 <= p <= mesh.m
        return (1.0/mesh.h) * (2.0*w[p] - w[p-1] - w[p+1]) - ell[p]

    def smoothersweep(self, mesh, w, ell, phi, forward=True):
        '''Do in-place projected Gauss-Seidel sweep, with relaxation factor
        omega, over the interior points p=1,...,m, for the classical obstacle
        problem
            F(u)[v-u] = a(w,v-u) - ell[v-u] >= 0
        for all v in V^j.  Input iterate w is in V^j and ell is in V^j'.
        At each p, solves
            F(w + c psi_p)[psi_p] = 0
        for c.  Thus c = - F(w)[psi_p] / a(psi_p,psi_p).  Update of w guarantees
        admissibility:
            w[p] <- max(w[p] + omega c, phi[p]).
        Input mesh is of class MeshLevel1D.  Returns the number of pointwise
        feasibility violations.'''
        infeascount = self._checkrepairadmissible(mesh, w, phi)
        for p in self._sweepindices(mesh, forward=forward):
            c = - self.pointresidual(mesh, w, ell, p) / _poissondiagonalentry(mesh, p)
            w[p] = max(w[p] + self.args.omega * c, phi[p])
        mesh.WU += 1
        return infeascount

    def phi(self, x):
        '''The obstacle:  u >= phi.'''
        if self.args.poissoncase == 'icelike':
            ph = x * (1.0 - x)
        elif self.args.poissoncase == 'parabola':
            # maximum is at  2.0 + args.parabolay
            ph = 8.0 * x * (1.0 - x) + self.args.parabolay
        else:
            raise ValueError
        if self.args.random:
            perturb = np.zeros(len(x))
            for jj in range(self.args.randommodes):
                perturb += np.random.randn(1) * np.sin((jj+1) * np.pi * x)
            perturb *= self.args.randomscale * 0.03 * np.exp(-10 * (x-0.5)**2)
            ph += perturb
        ph[[0, -1]] = [0.0, 0.0]  # always force zero boundary conditions
        return ph

    def exact_available(self):
        return (not self.args.random) and (self.args.fscale == 1.0) \
                 and (self.args.parabolay == -1.0 or self.args.parabolay <= -2.25)

    # **** problem-specific; other obstacle problems may not have these ****

    def source(self, x):
        '''The source term f in the interior condition -u'' = f.'''
        if self.args.poissoncase == 'icelike':
            f = 8.0 * np.ones(np.shape(x))
            f[x < 0.2] = -16.0
            f[x > 0.8] = -16.0
        else:
            f = -2.0 * np.ones(np.shape(x))
        return self.args.fscale * f

    def exact(self, x):
        '''Exact solution u(x), if available.  Assumes x is a numpy array.'''
        assert self.exact_available()
        if self.args.poissoncase == 'icelike':
            u = self.phi(x)
            a, c0, c1, d0, d1 = 0.1, -0.8, 0.09, 4.0, -0.39  # exact values
            mid = (x > 0.2) * (x < 0.8) # logical and
            left = (x > a) * (x < 0.2)
            right = (x > 0.8) * (x < 1.0-a)
            u[mid] = -4.0*x[mid]**2 + d0*x[mid] + d1
            u[left] = 8.0*x[left]**2 + c0*x[left] + c1
            u[right] = 8.0*(1.0-x[right])**2 + c0*(1.0-x[right]) + c1
        else:  # poissoncase == 'parabola'
            if self.args.parabolay == -1.0:
                a = 1.0/3.0
                def upoisson(x):
                    return x * (x - 18.0 * a + 8.0)
                u = self.phi(x)
                u[x < a] = upoisson(x[x < a])
                u[x > 1.0-a] = upoisson(1.0 - x[x > 1.0-a])
            elif self.args.parabolay <= -2.25:
                u = x * (x - 1.0)   # solution without obstruction
            else:
                raise NotImplementedError
        return u

class PJacobiPoisson(PGSPoisson):
    '''Derived class of PGSPoisson that replaces the Gauss-Seidel
    (multiplicative) smoother with a Jacobi (additive) version.'''

    def smoothersweep(self, mesh, w, ell, phi, forward=True):
        '''Do in-place projected Jacobi sweep, with relaxation factor
        omega, over the interior points p=1,...,m, for the classical obstacle
        problem.  Same as Gauss-Seidel but the new iterate values are NOT
        used when updating the next point; the residual is evaluated
        at the start and those values are used for the sweep.  Tai (2003)
        says underrelaxation is expected; omega = 0.8 seems to work.'''
        infeascount = self._checkrepairadmissible(mesh, w, phi)
        r = self.residual(mesh, w, ell)
        for p in self._sweepindices(mesh, forward=forward):
            c = - r[p] / _poissondiagonalentry(mesh, p)
            w[p] = max(w[p] + self.args.omega * c, phi[p])
        mesh.WU += 1
        return infeascount
