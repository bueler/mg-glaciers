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

# fine hats figure
plt.figure(figsize=(8,6))
xhat = 0.3
yhat = 0.0
x = np.linspace(0.0,1.0,11)
y = np.linspace(0.0,0.8,11)
zero = np.zeros(np.shape(x))
plt.plot(x,zero,'k',lw=2.0)
plt.plot(zero,y,'k',lw=2.0)
plt.plot(xhat,yhat,'k.',ms=16.0)
#circ = plt.Circle((xhat,yhat), 0.5, color='k', fill=False)
ax = plt.gca()
for ww in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]:
    th = 180.0
    if ww > 0.6:  # end arc at y-axis
        alpha = (180.0/np.pi) * np.arccos(xhat/(ww/2.0))
        th -= alpha
    arc = Arc((xhat,yhat), ww, ww, angle=0.0, theta1=0.0, theta2=th,
              color='k', lw=1.0)
    ax.add_patch(arc)
plt.text(0.85,0.6,r'$\mathcal{Q}$',fontsize=bigfsize)
plt.text(0.25,-0.13,r'$(\hat x_1,\hat x_2)$',fontsize=bigfsize)
plt.axis('off')
plt.axis('equal')
figsave('cartoonplane.pdf')

