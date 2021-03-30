'''Module for the ObstacleMonitor class.  Evaluates and prints residual norms
and numerical errors.'''

import numpy as np

__all__ = ['indentprint', 'ObstacleMonitor']

def indentprint(n, s):
    '''Print 2n spaces and then string s.'''
    for _ in range(n):
        print('  ', end='')
    print(s)

class ObstacleMonitor():
    '''The monitor has an internal state so it is a class.'''

    def __init__(self, obsprob, mesh, uex=None,
                 printresiduals=False, printerrors=False):
        self.obsprob = obsprob  # Class SmootherObstacleProblem
        self.mesh = mesh        # Class MeshLevel1D
        self.residuals = printresiduals
        self.errors = printerrors
        self.uex = uex
        self.lastirnorm = None
        self.s = 0

    def inactiveresidual(self, w, ell, phi, ireps=1.0e-10):
        '''Compute the values of the residual for w at nodes where the constraint
        is NOT active.  Note that where the constraint is active the residual F(w)
        in the complementarity problem is allowed to have any positive value, and
        only the residual at inactive nodes is relevant to convergence.'''
        F = self.obsprob.residual(self.mesh, w, ell)
        F[w < phi + ireps] = np.minimum(F[w < phi + ireps], 0.0)
        return F

    def irerr(self, w, ell, phi, indent=0):
        '''Report inactive residual norm and error if available.'''
        irnorm = self.mesh.l2norm(self.inactiveresidual(w, ell, phi))
        ind = indent * '  '
        if self.residuals:
            print(ind + '  %d:  |ir(u)|_2 = %.4e' % (self.s, irnorm), end='')
            if self.lastirnorm is not None and self.lastirnorm > 0.0:
                print('  (rate %.4f)' % (irnorm/self.lastirnorm))
            else:
                print()
            self.lastirnorm = irnorm
        errnorm = None
        if self.errors:
            if self.uex is not None:
                errnorm = self.mesh.l2norm(w - self.uex)
                print(ind + '  %d:  |u-uexact|_2 = %.4e' % (self.s, errnorm))
        self.s += 1
        return irnorm, errnorm
