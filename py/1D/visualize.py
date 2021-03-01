'''Module for visualizing obstacle problem results (class VisObstacle).'''

__all__ = ['VisObstacle']

import matplotlib
import matplotlib.pyplot as plt
from monitor import ObstacleMonitor

# better defaults for graphs
font = {'size' : 20}
matplotlib.rc('font', **font)
lines = {'linewidth': 2}
matplotlib.rc('lines', **lines)

def _output(filename, description):
    '''Either save result to an image file or use show().  Supply '' as filename
    to use show().'''
    if len(filename) == 0:
        plt.show()
    else:
        print('saving %s to %s ...' % (description, filename))
        plt.savefig(filename, bbox_inches='tight')

class VisObstacle():
    '''Class for visualizing obstacle problem results.
    Initialize with a mesh and an obstacle on that mesh.'''

    def __init__(self, args, obsprob, hierarchy, u=None, phi=None, ell=None, uex=None):
        self.args = args              # see obstacle.py for user options
        self.obsprob = obsprob        # Class SmootherObstacleProblem
        self.hierarchy = hierarchy    # array of Class MeshLevel1D
        self.mesh = hierarchy[-1]     # finest mesh
        self.monitor = ObstacleMonitor(self.obsprob, self.mesh)
        self.mesh.checklen(phi)
        self.u = u                    # final iterate (numerical solution)
        self.phi = phi                # obstacle
        self.ell = ell                # final source
        self.uex = uex                # exact solution

    def plain(self, filename=''):
        '''Generate plain graphic showing obstacle and exact solution
        (if available) on finest mesh.'''
        xx = self.mesh.xx()
        plt.figure(figsize=(16.0, 4.0))
        if self.uex is not None:
            self.mesh.checklen(self.uex)
            plt.plot(xx, self.uex, 'k--', label=r'solution $u$', lw=3.0)
        plt.plot(xx, self.phi, 'k', label=r'obstacle $\varphi$', lw=3.0)
        plt.legend()
        plt.xlabel('x')
        _output(filename, 'exact solution and obstacle')

    def final(self, filename=''):
        '''Generate graphic showing final iterate, obstacle, and exact
        solution (if available) on finest mesh.'''
        self.mesh.checklen(self.u)
        xx = self.mesh.xx()
        plt.figure(figsize=(15.0, 8.0))
        plt.plot(xx, self.u, 'k', label='final iterate', linewidth=4.0)
        if self.uex is not None:
            self.mesh.checklen(self.uex)
            plt.plot(xx, self.uex, 'g', label='exact solution')
        plt.plot(xx, self.phi, 'r', label='obstacle')
        plt.axis('tight')
        plt.legend()
        plt.xlabel('x')
        _output(filename, 'final iterate and obstacle')

    def residuals(self, filename=''):
        '''Generate graphic showing residual and inactive residual vectors.'''
        xx = self.mesh.xx()
        plt.figure(figsize=(15.0, 10.0))
        plt.subplot(2, 1, 1)
        r = self.obsprob.residual(self.mesh, self.u, self.ell)
        ir = self.monitor.inactiveresidual(self.u, self.ell, self.phi)
        plt.plot(xx, r, 'k', label='residual')
        plt.plot(xx, ir, 'r', label='inactive residual')
        plt.legend()
        plt.gca().set_xticks([], minor=False)
        plt.subplot(2, 1, 2)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(xx, ir, 'r', label='inactive residual')
        plt.legend()
        plt.xlabel('x')
        _output(filename, 'residual and inactive residual')

    def decomposition(self, up=0, filename=''):
        '''Hierarchical defect decomposition.'''
        assert self.hierarchy[-1].m == self.mesh.m
        plt.figure(figsize=(15.0, 10.0))
        J = len(self.hierarchy) - 1
        if up == 0:
            for j in range(J):
                plt.plot(self.hierarchy[j].xx(), self.hierarchy[j].chi,
                         'k.--', ms=10.0, label=r'$\chi^{%d}$' % j)
            plt.plot(self.mesh.xx(), self.mesh.chi, 'k.-',
                     ms=14.0, linewidth=3.0,
                     label=r'$\chi^{%d} = \varphi^{%d} - w^{%d}$' % (J, J, J))
        else:
            raise NotImplementedError('decomposition not available when up>1')
        plt.legend(fontsize=24.0, frameon=False)
        plt.xlabel('x')
        _output(filename, 'hierarchical decomposition')

    def icedecomposition(self, up=0, filename=''):
        '''Multilevel "ice-like" decomposition.'''
        assert self.hierarchy[-1].m == self.mesh.m
        plt.figure(figsize=(15.0, 10.0))
        J = len(self.hierarchy) - 1
        if up == 0:
            for j in range(J, -1, -1):
                z = self.hierarchy[j].chi
                for k in range(j, J):
                    z = self.hierarchy[k+1].cP(z)
                if j == J:
                    chilabel = r'$w^{%d}$' % J
                    chistyle = 'k'
                else:
                    chilabel = r'level $%d$' % j   # i.e. phi - chi^j
                    chistyle = 'k--'
                plt.plot(self.mesh.xx(), self.phi - z, chistyle,
                         label=chilabel)
        else:
            raise NotImplementedError('ice-like decomposition not available when up>1')
        plt.plot(self.mesh.xx(), self.phi, 'k',
                 label=r'$\varphi^{%d}$' % J, linewidth=4.0)
        plt.legend(fontsize=24.0, frameon=False)
        plt.xlabel('x')
        _output(filename, '"ice-like" decomposition')

    def generate(self):
        '''Generate figures according to arguments in args.'''
        if self.args.show or self.args.o:
            if self.args.plain:
                self.plain(filename=self.args.o)
            else:
                self.final(filename=self.args.o)
        if self.args.diagnostics:
            rname = ''
            if len(self.args.o) > 0:
                rname = 'resid_' + self.args.o
            self.residuals(filename=rname)
            if not self.args.pgsonly:
                dname, iname = '', ''
                if len(self.args.o) > 0:
                    dname = 'decomp_' + self.args.o
                    iname = 'icedec_' + self.args.o
                self.decomposition(up=self.args.up, filename=dname)
                self.icedecomposition(up=self.args.up, filename=iname)
