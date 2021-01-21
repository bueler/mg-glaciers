# module for visualizing obstacle problem results

import matplotlib
import matplotlib.pyplot as plt
from poisson import residual

__all__ = ['obstacleplot,obstaclediagnostics']

# better defaults for graphs
font = {'size' : 20}
matplotlib.rc('font', **font)
lines = {'linewidth': 2}
matplotlib.rc('lines', **lines)

def output(filename):
    if filename:
        print('saving output to %s ...' % filename)
        plt.savefig(filename,bbox_inches='tight')
    else:
        plt.show()

def inactiveresidual(mesh,w,ell,phi):
    '''Compute the values of the residual at nodes where the constraint
    is NOT active.  Note that where the constraint is active the residual
    may have significantly negative values.  The norm of the residual at
    inactive nodes is relevant to convergence.'''
    r = residual(mesh,w,ell)
    ireps = 1.0e-10
    r[w < phi + ireps] = np.maximum(r[w < phi + irreps],0.0)
    return r

def obstacleplot(mesh,uinitial,ufinal,phi,filename,uex=[]):
    xx = mesh.xx()
    plt.figure(figsize=(15.0,8.0))
    plt.plot(xx,uinitial,'k--',label='initial iterate')
    plt.plot(xx,ufinal,'k',label='final iterate',linewidth=4.0)
    plt.plot(xx,phi,'r',label='obstacle')
    if len(uex) > 0:
        plt.plot(xx,uex,'g',label='exact')
    plt.axis([0.0,1.0,-0.3 + min(ufinal),1.1*max(ufinal)])
    plt.legend()
    plt.xlabel('x')
    output(filename)

def obstaclediagnostics(hierarchy,ufinal,phi,ell,chi,filename):
    mesh = hierarchy[-1]
    xx = mesh.xx()
    plt.figure(figsize=(15.0,15.0))
    plt.subplot(4,1,1)
    r = residual(mesh,w,ell,phi)
    ir = inactiveresidual(mesh,w,ell,phi)
    plt.plot(xx,r,'k',label='residual')
    plt.plot(xx,ir,'r',label='inactive residual')
    plt.legend()
    plt.gca().set_xticks([],[])
    plt.subplot(4,1,2)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(xx,ir,'r',label='inactive residual')
    plt.legend()
    plt.gca().set_xticks([],[])
    plt.subplot(4,1,(3,4))
    for k in range(levels-1):
        plt.plot(hierarchy[k].xx(),chi[k],'k.--',ms=8.0,
                 label='level %d' % k)
    plt.plot(hierarchy[levels-1].xx(),chi[levels-1],'k.-',ms=12.0,
             label='fine mesh',linewidth=3.0)
    plt.legend()
    plt.title('decomposition of final defect obstacle')
    plt.xlabel('x')
    if filename:
        filename = 'diags_' + filename
    output(filename)

