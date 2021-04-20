'''Module for the ObstacleMonitor class.  Evaluates and prints residual norms
and numerical errors.'''

import numpy as np

__all__ = ['indentprint', 'ObstacleMonitor']

def indentprint(n, s, end='\n'):
    '''Print 2n spaces and then string s.'''
    for _ in range(n):
        print('  ', end='')
    print(s, end=end)

class ObstacleMonitor():
    '''The monitor has an internal state so it is a class.'''

    def __init__(self, obsprob, mesh, uex=None,
                 printresiduals=False, printerrors=False, l1err=False,
                 extraerrorpower=None, extraerrornorm=None):
        self.obsprob = obsprob  # Class SmootherObstacleProblem
        self.mesh = mesh        # Class MeshLevel1D
        self.residuals = printresiduals
        self.errors = printerrors
        self.l1err = l1err
        self.extraerrorpower = extraerrorpower
        self.extraerrornorm = extraerrornorm
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
        if self.residuals:
            indentprint(indent, '  %d:  |ir(u)|_2 = %.4e' % (self.s, irnorm), end='')
            if self.lastirnorm is not None and self.lastirnorm > 0.0:
                print('  (rate %.4f)' % (irnorm/self.lastirnorm))
            else:
                print()
            self.lastirnorm = irnorm
        errnorm = None
        if self.errors:
            if self.uex is not None:
                e2norm = self.mesh.l2norm(w - self.uex)
                if self.l1err:
                    e1norm = self.mesh.l1norm(w - self.uex)
                    indentprint(indent, '  %d:  |u-uexact|_1 = %.4e, |u-uexact|_2 = %.4e' \
                                % (self.s, e1norm, e2norm), end='')
                else:
                    indentprint(indent, '  %d:  |u-uexact|_2 = %.4e' \
                                % (self.s, e2norm), end='')
                if self.extraerrorpower is not None and self.extraerrornorm is not None:
                    r = self.extraerrorpower
                    p = self.extraerrornorm
                    print(', |u^r-uexact^r|_p = %.4e' % self.mesh.lqnorm(p, w**r - self.uex**r))
                else:
                    print()
        self.s += 1
        return irnorm, errnorm
