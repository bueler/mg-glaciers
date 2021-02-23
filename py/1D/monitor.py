'''Module for the ObstacleMonitor class suitable for obstacle problems.'''

from pgs import inactiveresidual

__all__ = ['ObstacleMonitor']

class ObstacleMonitor():
    '''The monitor has an internal state so it is a class.'''

    def __init__(self, mesh, ellfine, phifine, uex=None,
                 printresiduals=False, printerrors=False):
        self.mesh = mesh
        self.ell = ellfine
        self.phi = phifine
        self.uex = uex
        self.residuals = printresiduals
        self.errors = printerrors
        self.lastirnorm = None
        self.s = 0

    def irerr(self, w, indent=0):
        '''Report inactive residual norm and error if available.'''
        irnorm = self.mesh.l2norm(inactiveresidual(self.mesh, w, self.ell, self.phi))
        ind = indent * '  '
        if self.residuals:
            print(ind + '  %d:  |ir(u)|_2 = %.4e' % (self.s, irnorm), end='')
            if self.lastirnorm is not None and self.lastirnorm > 0.0:
                print('  (rate %.4f)' % (irnorm/self.lastirnorm))
            else:
                print()
            self.lastirnorm = irnorm
        if self.uex is not None:
            errnorm = self.mesh.l2norm(w-self.uex)
        else:
            errnorm = None
        if self.errors and self.uex is not None:
            print(ind + '  %d:  |u-uexact|_2 = %.4e' % (self.s, errnorm))
        self.s += 1
        return irnorm, errnorm
