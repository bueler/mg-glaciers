'''Module for visualizing obstacle problem results (class VisObstacle).'''

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

def _output(filename,description):
    '''Either save result to an image file or use show().  Supply '' as filename
    to use show().'''
    if len(filename) == 0:
        plt.show()
    else:
        print('saving %s to %s ...' % (description,filename))
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

    def plain(self, uex, filename=''):
        '''Generate plain graphic showing exact solution and obstacle.'''
        self.mesh.checklen(uex)
        xx = self.mesh.xx()
        plt.figure(figsize=(15.0, 6.0))
        plt.plot(xx, uex, 'k--', label=r'solution $u$', lw=3.0)
        plt.plot(xx, self.phi, 'k', label=r'obstacle $\varphi$', lw=3.0)
        plt.legend()
        plt.xlabel('x')
        _output(filename,'exact solution and obstacle')

    def final(self, ufinal, filename='', uex=None):
        '''Generate graphic showing final iterate, obstacle, and exact
        solution if given.'''
        self.mesh.checklen(ufinal)
        xx = self.mesh.xx()
        plt.figure(figsize=(15.0, 8.0))
        plt.plot(xx, ufinal, 'k', label='final iterate', linewidth=4.0)
        if uex is not None:
            self.mesh.checklen(uex)
            plt.plot(xx, uex, 'g', label='exact solution')
        plt.plot(xx, self.phi, 'r', label='obstacle')
        plt.axis('tight')
        plt.legend()
        plt.xlabel('x')
        _output(filename,'final iterate and obstacle')

    def residuals(self, ufinal, ell, filename=''):
        '''Generate graphic showing residual and inactive residual vectors.'''
        xx = self.mesh.xx()
        plt.figure(figsize=(15.0, 10.0))
        plt.subplot(2, 1, 1)
        r = residual(self.mesh, ufinal, ell)
        ir = inactiveresidual(self.mesh, ufinal, ell, self.phi)
        plt.plot(xx, r, 'k', label='residual')
        plt.plot(xx, ir, 'r', label='inactive residual')
        plt.legend()
        plt.gca().set_xticks([],minor=False)
        plt.subplot(2, 1, 2)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(xx, ir, 'r', label='inactive residual')
        plt.legend()
        plt.xlabel('x')
        _output(filename,'residual and inactive residual')

    def decomposition(self, hierarchy, up=0, filename=''):
        '''Hierarchical defect decomposition.'''
        assert hierarchy[-1].m == self.mesh.m
        plt.figure(figsize=(15.0, 10.0))
        J = len(hierarchy) - 1
        if up == 0:
            for j in range(J):
                plt.plot(hierarchy[j].xx(), hierarchy[j].chi, 'k.--', ms=10.0,
                         label=r'$\chi^{%d}$' % j)
            plt.plot(hierarchy[-1].xx(), hierarchy[-1].chi, 'k.-', ms=14.0, linewidth=3.0,
                     label=r'$\chi^{%d} = \varphi^{%d} - w^{%d}$' % (J,J,J))
        else:
            #FIXME
            raise NotImplementedError
        plt.legend(fontsize=24.0,frameon=False)
        #plt.title('decomposition of defect obstacle')
        plt.xlabel('x')
        _output(filename,'hierarchical decomposition')

    def icedecomposition(self, hierarchy, phi, up=0, filename=''):
        '''Multilevel "ice-like" decomposition.'''
        assert hierarchy[-1].m == self.mesh.m
        plt.figure(figsize=(15.0, 10.0))
        J = len(hierarchy) - 1
        if up == 0:
            for j in range(J,-1,-1):
                z = hierarchy[j].chi
                for k in range(j,J):
                    z = hierarchy[k+1].P(z)
                if j == J:
                    chilabel = r'$w^{%d}$' % J
                    chistyle = 'k'
                else:
                    chilabel = r'level $%d$' % j   # i.e. phi - chi^j
                    chistyle = 'k--'
                plt.plot(hierarchy[-1].xx(), phi - z, chistyle, label=chilabel)
        else:
            #FIXME
            raise NotImplementedError
        plt.plot(hierarchy[-1].xx(), phi, 'k', label=r'$\varphi^{%d}$' % J, linewidth=4.0)
        plt.legend(fontsize=24.0,frameon=False)
        #plt.title('"ice-like" multilevel decomposition')
        plt.xlabel('x')
        _output(filename,'"ice-like" decomposition')