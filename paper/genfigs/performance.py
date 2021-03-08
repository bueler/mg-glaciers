#!/usr/bin/env python3

# see py/1D/study/convperf.sh  to generate input data performance.txt
# FIXME only consider case = icelike+random (as more representative)
# FIXME process into two performance figures each showing all four solvers
#    Figure:  semilogx with m on x-axis and WU on y axis
#    Figure:  loglog with m on x-axis and time on y axis;  result is time = O(m^xx)

import numpy as np
import matplotlib.pyplot as plt

INFILE = 'performance.txt'
SOLVERNAME = ['V(1,0)', 'V(2,0)', 'NI+V(1,0)', 'NIx2+V(1,0)']
MARKER = ['o','s','*','p']
MARKERSIZE = [12.0,10.0,12.0,10.0]
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
solver = v[case==1,0]
m = v[case==1,3]
h = 1.0 / (m+1)
cycles = v[case==1,4]
WU = v[case==1,5]
time = v[case==1,6]

# work units versus m
plt.figure(figsize=(7,6))
for s in range(4):  # s=solver
    mm = m[solver==s]
    WUU = WU[solver==s]
    q = np.polyfit(np.log(mm),np.log(WUU),1)
    print('%15s: WU = O(m^%.2f)' % (SOLVERNAME[s],q[0]))
    plt.semilogx(mm,WUU,
                 'k'+MARKER[s],ms=MARKERSIZE[s],mfc=MARKERFACE[s],
                 label=SOLVERNAME[s] + ':  $O(m^{%.2f})$' % q[0])
plt.grid(True)
plt.xlabel('$m$  (degrees of freedom)',fontsize=18.0)
plt.ylabel('WU',fontsize=18.0)
plt.xlim([100.0,1.0e5])
plt.ylim([0.0,100.0])
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.legend(loc='lower right',fontsize=14.0)
writeout('mcdl-wu.pdf')

# (time/m) versus m
plt.figure(figsize=(7,6))
for s in range(4):  # s=solver
    timeper = time[solver==s] / mm
    mmhigh = mm[mm > 4000]
    timeperhigh = timeper[mm > 4000]
    q = np.polyfit(np.log(mmhigh),np.log(timeperhigh),1)
    print('%15s: time/m = O(m^%.2f)' % (SOLVERNAME[s],q[0]))
    plt.semilogx(mm,timeper*1000.0,
               'k'+MARKER[s],ms=MARKERSIZE[s],mfc=MARKERFACE[s],
               label=SOLVERNAME[s] + ':  $O(m^{%.2f})$' % q[0])
    plt.semilogx(mmhigh,1000.0*np.exp(q[0]*np.log(mmhigh) + q[1]),'k--',lw=0.5)
plt.grid(True)
plt.xlabel('$m$  (degrees of freedom)',fontsize=18.0)
plt.ylabel('time (ms) per $m$',fontsize=18.0)
plt.xlim([100.0,1.0e5])
plt.ylim([0.0,4.5])
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.legend(loc='upper right',fontsize=14.0)
writeout('mcdl-timeper.pdf')
