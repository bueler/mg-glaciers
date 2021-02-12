#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

fsize=16.0
bigfsize=18.0

debug = False
def figsave(name):
    if debug:
        plt.show()  # debug
    else:
        print('saving %s ...' % name)
        plt.savefig(name,bbox_inches='tight',transparent=True)

# fine mesh
m = 9   # 10 elements
x = np.linspace(0.0,1.0,m+2)
np.random.seed(2)
y = 1.5*x*(1-x) + 0.05 * np.random.randn(m+2)
y[0] = 0.0
y[m+1] = 0.0
y[1] = y[1] + 0.2
y[5] = y[5] - 0.15
y[6] = y[6] - 0.1
y[9] = y[9] - 0.05

# coarse mesh
xc = x[::2]
yc = y[::2]

# admissible iterate on coarse
w = yc.copy()
w[1] = w[1] + 0.13
w[2] = w[2] + 0.2
w[4] = w[4] + 0.1

# generate figure
plt.figure(figsize=(12,5))
plt.plot(xc, w,  'k--', lw=2.0, label='admissible iterate (coarse)')
plt.plot(xc, yc, 'k.-', lw=3.0, ms=14.0, label='coarse obstacle')
plt.plot(x,  y,  'k',   lw=1.5, label='fine obstacle')
plt.legend(fontsize=bigfsize,loc='upper right',frameon=False)
plt.axis([-0.01,1.2,-0.01,1.3*max(y)])
plt.axis('off')
figsave('prolongobstacle.pdf')

