'''Module for GS and Jacobi smoothers for the Poisson (classical) obstacle problem.'''

__all__ = ['PsmootherPoisson']

import numpy as np
from smoothers.base import SmootherObstacleProblem

def _pde2alpha(x):
    return 2.0 + np.sin(2.0 * np.pi * x)

class PsmootherPoisson(SmootherObstacleProblem):
    '''Class for the projected Gauss-Seidel (PGS) algorithm as a smoother
    for the linear Poisson equation - (alpha u')' = f with u(0)=u(L)=0.'''

    def __init__(self, args, admissibleeps=1.0e-10):
        super().__init__(args, admissibleeps=admissibleeps)
        # fix the random seed for repeatability
        np.random.seed(self.args.randomseed)
        if self.args.poissoncase == 'pde2':
            self.alpha = _pde2alpha
        else:
            self.alpha = None
        # smoother name
        self.name = 'PJac' if self.args.jacobi else 'PGS'

    def _diagonalentry(self, h, p):
        '''Compute the diagonal value of a(.,.) at hat function psi_p^j:
           a(psi_p,psi_p) = int_0^L alpha(x) (psi_p^j)'(x)^2 dx
        Uses trapezoid rule if alpha(x) not constant.'''
        if self.alpha is not None:
            xx = h * np.array([p-1,p,p+1])
            aa = self.alpha(xx)
            return (1.0 / (2.0 * h)) * (aa[0] + 2.0 * aa[1] + aa[2])
        else:
            return 2.0 / h

    def _pointresidual(self, h, w, ell, p):
        '''Compute the value of the residual linear functional, in V^j', for given
        iterate w, at one interior hat function psi_p^j:
           F(w)[psi_p^j] = int_0^L alpha(x) w'(x) (psi_p^j)'(x) dx - ell(psi_p^j)
        Uses trapezoid rule if alpha(x) not constant.  Input ell is in V^j'.'''
        if self.alpha is not None:
            xx = h * np.array([p-1,p,p+1])
            aa = self.alpha(xx)
            zz = (aa[0] + aa[1]) * (w[p] - w[p-1]) \
                 - (aa[1] + aa[2]) * (w[p+1] - w[p])
            return (1.0 / (2.0 * h)) * zz - ell[p]
        else:
            return (1.0 / h) * (2.0*w[p] - w[p-1] - w[p+1]) - ell[p]

    def applyoperator(self, mesh, w):
        raise NotImplementedError

    def residual(self, mesh, w, ell):
        '''Compute the residual functional for given iterate w.  Note
        ell is a source term in V^j'.  Calls _pointresidual() for values.'''
        mesh.checklen(w)
        mesh.checklen(ell)
        F = mesh.zeros()
        for p in range(1, mesh.m+1):
            F[p] = self._pointresidual(mesh.h, w, ell, p)
        return F

    def smoothersweep(self, mesh, w, ell, phi, forward=True):
        '''Do either in-place GS or Jacobi smoothing.'''
        mesh.checklen(w)
        mesh.checklen(ell)
        mesh.checklen(phi)
        self._checkrepairadmissible(mesh, w, phi)
        if self.args.jacobi:
            self.jacobisweep(mesh, w, ell, phi, forward=forward)
        else:
            self.gssweep(mesh, w, ell, phi, forward=forward)
        mesh.WU += 1

    def gssweep(self, mesh, w, ell, phi, forward=True):
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
        Input mesh is of class MeshLevel1D.'''
        for p in self._sweepindices(mesh, forward=forward):
            c = - self._pointresidual(mesh.h, w, ell, p) / self._diagonalentry(mesh.h, p)
            w[p] = max(w[p] + self.args.omega * c, phi[p])

    def jacobisweep(self, mesh, w, ell, phi, forward=True):
        '''Do in-place projected Jacobi sweep, with relaxation factor
        omega, over the interior points p=1,...,m, for the classical obstacle
        problem.  Same as Gauss-Seidel but the new iterate values are NOT
        used when updating the next point; the residual is evaluated
        at the start and those values are used for the sweep.  Tai (2003)
        says underrelaxation is expected; omega = 0.8 seems to work.'''
        r = self.residual(mesh, w, ell)
        for p in self._sweepindices(mesh, forward=forward):
            c = - r[p] / self._diagonalentry(mesh.h, p)
            w[p] = max(w[p] + self.args.omega * c, phi[p])

    def phi(self, x):
        '''The obstacle:  u >= phi.'''
        if self.args.poissoncase == 'icelike':
            ph = x * (1.0 - x)
        elif self.args.poissoncase == 'traditional':
            # maximum is at  2.0 + args.poissonparabolay
            ph = 8.0 * x * (1.0 - x) + self.args.poissonparabolay
        elif self.args.poissoncase in ['pde1', 'pde2']:  # these u(x) are above axis
            ph = - np.ones(np.shape(x))
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
        return (not self.args.random) and (self.args.poissonfscale == 1.0) \
                 and (self.args.poissonparabolay == -1.0 or self.args.poissonparabolay <= -2.25)

    def source(self, x):
        '''The source term f in the interior condition - (alpha u')' = f.'''
        if self.args.poissoncase == 'icelike':
            f = 8.0 * np.ones(np.shape(x))
            f[x < 0.2] = -16.0
            f[x > 0.8] = -16.0
        elif self.args.poissoncase == 'traditional':
            f = -2.0 * np.ones(np.shape(x))
        elif self.args.poissoncase == 'pde1':
            f = 12.0 * x * x - 2.0
        elif self.args.poissoncase == 'pde2':
            twopi = 2.0 * np.pi
            f = - twopi * (10.0 - 2.0 * x) * np.cos(twopi * x) \
                + 2.0 * (2.0 + np.sin(twopi * x))
        else:
            raise ValueError
        return self.args.poissonfscale * f

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
        elif self.args.poissoncase == 'traditional':
            if self.args.poissonparabolay == -1.0:
                a = 1.0/3.0
                def upoisson(x):
                    return x * (x - 18.0 * a + 8.0)
                u = self.phi(x)
                u[x < a] = upoisson(x[x < a])
                u[x > 1.0-a] = upoisson(1.0 - x[x > 1.0-a])
            elif self.args.poissonparabolay <= -2.25:
                u = x * (x - 1.0)   # solution without obstruction
            else:
                raise NotImplementedError
        elif self.args.poissoncase == 'pde1':
            u = x * x * (1.0 - x * x)
        elif self.args.poissoncase == 'pde2':
            u = x * (10.0 - x)
        return u
