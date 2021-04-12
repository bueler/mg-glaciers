#!/usr/bin/env python3

# see py/1D/study/siaconv.sh  to generate input data  siaconv.txt

import numpy as np
import matplotlib.pyplot as plt

INFILE = 'siaconv.txt'
MARKER = ['o','o']
MARKERSIZE = [10.0,10.0]
MARKERFACE = ['w','k']
NAME = ['$|s-s_{exact}|_2$','$|H^r-H_{exact}^r|_p$']

SHOW = False
def writeout(outname):
    if SHOW:
        plt.show()
    else:
        print('writing file ' + outname)
        plt.savefig(outname,bbox_inches='tight')

print('reading %s ...' % INFILE)
v = np.array(np.loadtxt(INFILE))
#print(v)

N = 13  # number of different resolutions
assert len(v[:,0]) == N

# columns:  J m err2 errrp
L = 1800.0e3  # m
m = v[:,1]
h = L / (m+1)
err2 = v[:,2]
errrp = v[:,3]

# numerical error versus h
plt.figure(figsize=(7,6))
q2 = np.polyfit(np.log(h),np.log(err2),1)
qrp = np.polyfit(np.log(h),np.log(errrp),1)
print('|s-sexact|_2 = O(h^%.2f), |H^r - Hexact^r|_p = O(h^%.2f)' \
      % (q2[0],qrp[0]))
plt.loglog(h,err2,
           'k'+MARKER[0],ms=MARKERSIZE[0],mfc=MARKERFACE[0],
           label=NAME[0] + '$=O(h^{%.2f})$' % q2[0])
plt.loglog(h,np.exp(q2[0]*np.log(h)+q2[1]),'k--')
plt.loglog(h,errrp,
           'k'+MARKER[1],ms=MARKERSIZE[1],mfc=MARKERFACE[1],
           label=NAME[1] + '$=O(h^{%.2f})$' % qrp[0])
plt.loglog(h,np.exp(qrp[0]*np.log(h)+qrp[1]),'k--')
plt.grid(True)
plt.axis([10.0,1.0e6,1.0e2,1.0e11])
plt.xlabel('$h$  (m)',fontsize=18.0)
plt.ylabel('numerical error',fontsize=18.0)
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.legend(loc='upper left',fontsize=14.0)
writeout('siaconv.pdf')
