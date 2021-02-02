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
        '''Generate plain graphic showing exact solution and obstacle in
        black.'''
        self.mesh.checklen(uex)
        xx = self.mesh.xx()
        plt.figure(figsize=(15.0, 6.0))
        plt.plot(xx, uex, 'k--', label=r'solution $u$', lw=3.0)
        plt.plot(xx, self.phi, 'k', label=r'obstacle $\varphi$', lw=3.0)
        plt.axis('tight')
        plt.axis('off')
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

    def decomposition(self, hierarchy, chi, up=0, filename=''):
        '''Hierarchical defect decomposition.'''
        assert hierarchy[-1].m == self.mesh.m
        plt.figure(figsize=(15.0, 10.0))
        K = len(hierarchy) - 1
        if up == 0:
            for k in range(K):
                plt.plot(hierarchy[k].xx(), chi[k], 'k.--', ms=10.0,
                         label=r'$\chi^%d$' % k)
            plt.plot(hierarchy[-1].xx(), chi[-1], 'k.-', ms=14.0,
                     label=r'$\chi^%d = \varphi^%d - w$' % (K,K), linewidth=3.0)
        else:
            #FIXME
            raise NotImplementedError
        plt.legend(fontsize=24.0)
        #plt.title('decomposition of defect obstacle')
        plt.xlabel('x')
        _output(filename,'hierarchical decomposition')

    def icedecomposition(self, hierarchy, chi, phi, up=0, filename=''):
        '''Multilevel "ice-like" decomposition.'''
        assert hierarchy[-1].m == self.mesh.m
        plt.figure(figsize=(15.0, 10.0))
        K = len(hierarchy) - 1
        if up == 0:
            for k in range(K,-1,-1):
                z = chi[k]
                for j in range(k,K):
                    z = hierarchy[j+1].P(z)
                if k == K:
                    chilabel = r'$w$'
                    chistyle = 'k'
                else:
                    chilabel = r'level $%d$' % k   # i.e. phi - chi^k
                    chistyle = 'k--'
                plt.plot(hierarchy[-1].xx(), phi - z, chistyle, label=chilabel)
        else:
            #FIXME
            raise NotImplementedError
        plt.plot(hierarchy[-1].xx(), phi, 'k', label=r'$\varphi^%d$' % K, linewidth=4.0)
        plt.legend(fontsize=24.0)
        #plt.title('"ice-like" multilevel decomposition')
        plt.xlabel('x')
        _output(filename,'"ice-like" decomposition')
