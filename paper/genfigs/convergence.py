#!/usr/bin/env python3

# see py/1D/study/convperf.sh  to generate input data  convergence.txt

import numpy as np
import matplotlib.pyplot as plt

INFILE = 'convergence.txt'
MARKER = ['o','s']
MARKERSIZE = [12.0,10.0]
MARKERFACE = ['w','k']
PROBLEMNAME = ['ice-like','parabola']

SHOW = False
def writeout(outname):
    if SHOW:
        plt.show()
    else:
        print('writing file ' + outname)
        plt.savefig(outname,bbox_inches='tight')

print('reading %s ...' % INFILE)
v = np.array(np.loadtxt(INFILE))

N = 13  # number of different resolutions
assert len(v[:,0]) == 2*N

# columns:  problem J m cycles err
prob = v[:,0]
m = v[:,2]
h = 1.0 / (m+1)
err = v[:,4]

# numerical error versus h
plt.figure(figsize=(7,6))
for p in range(2):
    hh = h[prob==p]
    ee = err[prob==p]
    q = np.polyfit(np.log(hh),np.log(ee),1)
    print('%10s: O(h^%.2f)' % (PROBLEMNAME[p],q[0]))
    plt.loglog(hh,ee,
               'k'+MARKER[p],ms=MARKERSIZE[p],mfc=MARKERFACE[p],
               label=PROBLEMNAME[p] + ' $O(h^{%.2f})$' % q[0])
    plt.loglog(hh,np.exp(q[0]*np.log(hh) + q[1]),'k--',lw=0.5)
plt.grid(True)
plt.xlabel('h',fontsize=18.0)
plt.ylabel(r'$\|u-u_{exact}\|_2$',fontsize=18.0)
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.legend(loc='lower right',fontsize=14.0)
writeout('convergence.pdf')
