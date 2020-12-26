#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

fsize=16.0
bigfsize=20.0

debug = False
def figsave(name):
    if debug:
        plt.show()  # debug
    else:
        print('saving %s ...' % name)
        plt.savefig(name,bbox_inches='tight',transparent=True)

def finemeshaxes(m):
    x = np.linspace(0.0,1.0,2)
    y = np.linspace(0.0,1.0,2)
    xdots = np.linspace(0.0,1.0,m+1)
    np.random.seed(1)
    xdots[1:m] += 0.012 * np.random.randn(m-1)
    plt.plot(x, np.zeros(2), 'k', lw=1.0)
    plt.plot(np.zeros(2), y, 'k', lw=1.0)
    plt.text(-0.01,-0.2,'0',fontsize=fsize)
    plt.text(0.99,-0.2,'1',fontsize=fsize)
    return xdots

def hats(xdots,style='k',width=2.0,dotstyle='k.',dotsize=18.0):
    for k in range(len(xdots)-2):
        plt.plot(xdots[k:k+3],[0.0,1.0,0.0],style,lw=width)
    plt.plot(xdots[1:-1], np.zeros(len(xdots)-2),dotstyle,ms=dotsize)

m = 12

# fine hats figure
plt.figure(figsize=(10,4))
xdots = finemeshaxes(m)
hats(xdots)
plt.text(0.5,-0.3,r'$p$',fontsize=bigfsize)
plt.text(-0.05,0.0,'0',fontsize=fsize)
plt.text(-0.05,0.95,'1',fontsize=fsize)
plt.axis([-0.2,1.2,-0.4,1.2])
plt.axis('off')
figsave('finehats.pdf')

# coarse hats figure
plt.figure(figsize=(10,4))
xdots = finemeshaxes(m)
hats(xdots,style='k:',width=1.0,dotsize=14.0)
hats(xdots[0::2],width=2.5,dotstyle='ks',dotsize=12.0)
plt.axis([-0.2,1.2,-0.4,1.2])
plt.axis('off')
figsave('coarsehats.pdf')

# coarsest hats figure
plt.figure(figsize=(10,4))
xdots = finemeshaxes(m)
hats(xdots,style='k:',width=1.0,dotsize=14.0)
hats(xdots[0::4],width=2.5,dotstyle='kd',dotsize=16.0)
plt.axis([-0.2,1.2,-0.4,1.2])
plt.axis('off')
figsave('coarsesthats.pdf')

