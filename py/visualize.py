'''Module for visualizing obstacle problem results.  Defines class
VisObstacle.'''

__all__ = ['VisObstacle']

import matplotlib
import matplotlib.pyplot as plt
from poisson import residual
from pgs import inactiveresidual


# better defaults for graphs
font = {'size' : 20}
matplotlib.rc('font', **font)
lines = {'linewidth': 2}
matplotlib.rc('lines', **lines)

def _output(filename):
    '''Either save result to an image file or use show().  Supply '' as filename
    to use show().'''
    if len(filename) == 0:
        plt.show()
    else:
        print('saving output to %s ...' % filename)
        plt.savefig(filename, bbox_inches='tight')

class VisObstacle():
    '''Specialized class for visualizing obstacle problem results.
    Initialize with a mesh and an obstacle on that mesh.'''

    def __init__(self, mesh, phi):
        self.mesh = mesh
        self.mesh.checklen(phi)
        self.phi = phi
        self.hierarchy = None
        self.chi = None

    def initialfinal(self, uinitial, ufinal, filename='', uex=None):
        '''Generate graphic showing initial and final solution iterates and
        obstacle.  Show exact solution if given.'''
        self.mesh.checklen(uinitial)
        self.mesh.checklen(ufinal)
        xx = self.mesh.xx()
        plt.figure(figsize=(15.0, 8.0))
        plt.plot(xx, uinitial, 'k--', label='initial iterate')
        plt.plot(xx, ufinal, 'k', label='final iterate', linewidth=4.0)
        plt.plot(xx, self.phi, 'r', label='obstacle')
        if uex is not None:
            self.mesh.checklen(uex)
            plt.plot(xx, uex, 'g', label='exact')
        plt.axis([0.0, 1.0, -0.3 + min(ufinal), 1.1*max(ufinal)])
        plt.legend()
        plt.xlabel('x')
        _output(filename)

    def decomposition(self, hierarchy, chi):
        '''Set the subspace decomposition.'''
        self.hierarchy = hierarchy
        assert hierarchy[-1].m == self.mesh.m
        self.chi = chi

    def _plotdecomposition(self, up=0):
        if up == 0:
            for k in range(len(self.hierarchy)-1):
                plt.plot(self.hierarchy[k].xx(), self.chi[k], 'k.--', ms=8.0,
                         label='level %d' % k)
            plt.plot(self.hierarchy[-1].xx(), self.chi[-1], 'k.-', ms=12.0,
                     label='fine mesh', linewidth=3.0)
        else:
            #FIXME
            raise NotImplementedError
        plt.legend()
        plt.title('decomposition of final defect obstacle')
        plt.xlabel('x')

    def diagnostics(self, ufinal, ell, up=0, filename=''):
        '''Generate graphic showing residual and inactive residual vectors
        and hierarchical defect decomposition.'''
        assert self.hierarchy is not None
        assert self.chi is not None
        xx = self.mesh.xx()
        plt.figure(figsize=(15.0, 15.0))
        plt.subplot(4, 1, 1)
        r = residual(self.mesh, ufinal, ell)
        ir = inactiveresidual(self.mesh, ufinal, ell, self.phi)
        plt.plot(xx, r, 'k', label='residual')
        plt.plot(xx, ir, 'r', label='inactive residual')
        plt.legend()
        plt.gca().set_xticks([], [])
        plt.subplot(4, 1, 2)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(xx, ir, 'r', label='inactive residual')
        plt.legend()
        plt.gca().set_xticks([], [])
        plt.subplot(4, 1, (3, 4))
        self._plotdecomposition(up=up)
        _output(filename)
