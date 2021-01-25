'''Module for visualizing obstacle problem results.'''

import matplotlib
import matplotlib.pyplot as plt
from poisson import residual
from pgs import inactiveresidual

__all__ = ['obstacleplot', 'obstaclediagnostics']

# better defaults for graphs
font = {'size' : 20}
matplotlib.rc('font', **font)
lines = {'linewidth': 2}
matplotlib.rc('lines', **lines)

def output(filename):
    '''Either save result to an image file or use show().'''
    if filename:
        print('saving output to %s ...' % filename)
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

def obstacleplot(mesh, uinitial, ufinal, phi, filename, uex=None):
    '''Generate graphic showing initial and final solution iterates and
    obstacle.  Show exact solution if given.'''
    xx = mesh.xx()
    plt.figure(figsize=(15.0, 8.0))
    plt.plot(xx, uinitial, 'k--', label='initial iterate')
    plt.plot(xx, ufinal, 'k', label='final iterate', linewidth=4.0)
    plt.plot(xx, phi, 'r', label='obstacle')
    if uex is not None:
        plt.plot(xx, uex, 'g', label='exact')
    plt.axis([0.0, 1.0, -0.3 + min(ufinal), 1.1*max(ufinal)])
    plt.legend()
    plt.xlabel('x')
    output(filename)

def obstaclediagnostics(hierarchy, ufinal, phi, ell, chi, up, filename):
    '''Generate graphic showing residual and inactive residual vectors
    and hierarchical defect decomposition.'''
    mesh = hierarchy[-1]
    xx = mesh.xx()
    plt.figure(figsize=(15.0, 15.0))
    plt.subplot(4, 1, 1)
    r = residual(mesh, ufinal, ell)
    ir = inactiveresidual(mesh, ufinal, ell, phi)
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
    if up == 0:
        for k in range(len(hierarchy)-1):
            plt.plot(hierarchy[k].xx(), chi[k], 'k.--', ms=8.0,
                     label='level %d' % k)
        plt.plot(hierarchy[-1].xx(), chi[-1], 'k.-', ms=12.0,
                 label='fine mesh', linewidth=3.0)
    else:
        #FIXME
        raise NotImplementedError
    plt.legend()
    plt.title('decomposition of final defect obstacle')
    plt.xlabel('x')
    if filename:
        filename = 'diags_' + filename
    output(filename)
