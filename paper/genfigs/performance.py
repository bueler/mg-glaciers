#!/usr/bin/env python3

# see py/1D/study/convperf.sh  to generate input data performance.txt
# FIXME only consider case = icelike+random (as more representative)
# FIXME process into two performance figures each showing all four solvers
#    Figure:  semilogx with N on x-axis and WU on y axis
#    Figure:  loglog with N on x-axis and time on y axis;  result is time = O(N^xx)

import numpy as np
import matplotlib.pyplot as plt

INFILE = 'performance.txt'
SOLVENAME = ['V(1,0)', 'V(2,0)', 'NI+V(1,0)', 'NIx2+V(1,0)']
MARKER = ['o','s','*','+']
MARKERSIZE = [12.0,10.0,12.0,12.0]
MARKERFACE = ['w','w','k','k']

SHOW = False
def writeout(outname):
    if SHOW:
        plt.show()
    else:
        print('writing file ' + outname)
        plt.savefig(outname,bbox_inches='tight')

print('reading %s ...' % INFILE)
v = np.array(np.loadtxt(INFILE))

N = 10  # number of different resolutions
assert len(v[:,0]) == 8*N   # 8 solver+problem combinations

# columns:  solver case J m cycles WU time
case = v[:,1]
m = v[case==1,3]
h = 1.0 / (m+1)
cycles = v[case==1,4]
WU = v[case==1,5]
time = v[case==1,6]

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
plt.ylabel(r'$\|u^h-u\|_2$',fontsize=18.0)
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.legend(loc='lower right',fontsize=14.0)
writeout('convergence.pdf')
