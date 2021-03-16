#!/usr/bin/env python3

# see py/1D/study/convperf.sh  to generate input data  convergence.txt

import numpy as np
import matplotlib.pyplot as plt

INFILE = 'convergence.txt'
MARKER = ['o','s','p']
MARKERSIZE = [12.0,10.0,10.0]
MARKERFACE = ['w','k','w']
PROBLEMNAME = ['ice-like','traditional','unconstrained']

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
assert len(v[:,0]) == 3*N  # 3 problems, N resolutions

# columns:  problem J m cycles err
prob = v[:,0]
m = v[:,2]
h = 1.0 / (m+1)
err = v[:,4]

# run to get values for py/1D/study/perfni.sh:
#for j in range(2*N):
#    print('%.5e' % (2.0*err[j]))

# numerical error versus h
plt.figure(figsize=(7,6))
for p in range(3):
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
plt.ylabel(r'$\|u^h-u\|_2$',fontsize=18.0)
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.legend(loc='lower right',fontsize=14.0)
writeout('convergence.pdf')
