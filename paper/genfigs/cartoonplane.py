#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

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

plt.figure(figsize=(8,6))
xhat = 0.3
yhat = 0.0
yc = -0.3
x = np.linspace(0.0,1.1,11)
y = np.linspace(0.0,0.8,11)
zero = np.zeros(np.shape(x))
plt.plot(x,zero,'k',lw=2.0)
plt.plot(zero,y,'k',lw=2.0)
plt.plot(xhat,yhat,'k.',ms=12.0)
#circ = plt.Circle((xhat,yhat), 0.5, color='k', fill=False)
ax = plt.gca()
for r in [0.4, 0.5, 0.6, 0.7, 0.8]:
    alpha = (180.0/np.pi) * np.arcsin(-yc/r)
    beta = 90.0 + (180.0/np.pi) * np.arcsin(xhat/r)
    beta = min(beta,180.0 - alpha)
    arc = Arc((xhat,yc), 2.0*r, 2.0*r, angle=0.0, theta1=alpha, theta2=beta,
              color='k', lw=0.75)
    ax.add_patch(arc)
plt.text(0.85,0.6,r'$\mathcal{Q}$',fontsize=fsize)
plt.text(xhat-0.03,-0.13,r'$\hat x$',fontsize=fsize)
plt.axis('off')
plt.axis('equal')
figsave('cartoonplane.pdf')

