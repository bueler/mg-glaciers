#!/usr/bin/env python3

# see py/1D/study/performance.sh  to generate input data performance.txt
# here we only consider icelike case

# processed into three performance figures each showing four solvers
#    Figure:  semilogx with m on x-axis and number of cycles on y axis
#    Figure:  semilogx with m on x-axis and WU on y axis
#    Figure:  loglog with m on x-axis and time on y axis

import numpy as np
import matplotlib.pyplot as plt

INFILE = 'performance.txt'
SOLVERNAME = ['V(1,0)', 'V(0,1)', 'V(1,1)', 'V(1,1)-J']
MARKER = ['o','s','v','D']
MARKERSIZE = [12.0,11.0,10.0,9.0]
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
assert len(v[:,0]) == 4*N   # 4 solver+problem combinations

# columns:  solver J m cycles WU time
solver = v[:,0]
m = v[:,2]
h = 1.0 / (m+1)
cycles = v[:,3]
WU = v[:,4]
time = v[:,5]

# cycles versus m
plt.figure(figsize=(7,6))
for s in range(4):  # s=solver
    mm = m[solver==s]
    ccycles = cycles[solver==s]
    q = np.polyfit(np.log(mm),np.log(ccycles),1)
    print('%15s: cycles = O(m^%.2f)' % (SOLVERNAME[s],q[0]))
    plt.semilogx(mm,ccycles,'k'+MARKER[s],ms=MARKERSIZE[s],mfc=MARKERFACE[s],
                 label=SOLVERNAME[s] + ':  $O(m^{%.2f})$' % q[0])
plt.grid(True)
plt.xlabel('$m$',fontsize=18.0)
plt.ylabel('iterations',fontsize=18.0)
plt.axis([100.0,1.0e5,0.0,55.0])
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.legend(loc='lower right',fontsize=14.0)
writeout('mcdl-cycles.pdf')

# work units versus m
plt.figure(figsize=(7,6))
for s in range(4):  # s=solver
    mm = m[solver==s]
    WUU = WU[solver==s]
    q = np.polyfit(np.log(mm),np.log(WUU),1)
    print('%15s: WU = O(m^%.2f)' % (SOLVERNAME[s],q[0]))
    plt.semilogx(mm,WUU,'k'+MARKER[s],ms=MARKERSIZE[s],mfc=MARKERFACE[s],
                 label=SOLVERNAME[s] + ':  $O(m^{%.2f})$' % q[0])
plt.grid(True)
plt.xlabel('$m$',fontsize=18.0)
plt.ylabel('WU',fontsize=18.0)
plt.axis([100.0,1.0e5,0.0,210.0])
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.legend(loc='upper left',fontsize=14.0)
writeout('mcdl-wu.pdf')

# (time/m) versus m; skip three coarsest
plt.figure(figsize=(7,6))
for s in range(4):  # s=solver
    mm = m[solver==s]
    timeper = time[solver==s] / mm
    mmall = mm[mm > 1000]
    timeperall = timeper[mm > 1000]
    mmhigh = mm[mm > 4000]
    timeperhigh = timeper[mm > 4000]
    q = np.polyfit(np.log(mmhigh),np.log(timeperhigh),1)
    print('%15s: time/m = O(m^%.2f)' % (SOLVERNAME[s],q[0]))
    plt.semilogx(mmall,timeperall*1000.0,
               'k'+MARKER[s],ms=MARKERSIZE[s],mfc=MARKERFACE[s],
               label=SOLVERNAME[s] + ':  $O(m^{%.2f})$' % q[0])
    plt.semilogx(mmhigh,1000.0*np.exp(q[0]*np.log(mmhigh) + q[1]),'k--',lw=0.5)
plt.grid(True)
plt.xlabel('$m$',fontsize=18.0)
plt.ylabel('time (ms) per $m$',fontsize=18.0)
plt.axis([900.0,1.0e5,0.0,1.5])
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.legend(loc='lower right',fontsize=14.0)
writeout('mcdl-timeper.pdf')
