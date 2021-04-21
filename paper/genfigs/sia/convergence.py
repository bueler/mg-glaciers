#!/usr/bin/env python3

# see py/1D/study/siaconv.sh  to generate input data  siaconv.txt

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

INFILE = 'convergence.txt'
MARKER = ['o','s','D']
MARKERSIZE = [10.0,9.0,8.0]
MARKERFACE = ['w','w','k']
NAME = ['$|s-s_{exact}|_1$','$|s-s_{exact}|_2$','$|H^r-H_{exact}^r|_p$']

print('reading %s ...' % INFILE)
v = np.array(np.loadtxt(INFILE))
#print(v)

N = 13  # number of different resolutions
assert len(v[:,0]) == N

# columns:  J m err2 errrp
L = 1800.0e3  # m
m = v[:,1]
h = L / (m+1)
h /= 1000.0   # show in km
err1 = v[:,2]
err2 = v[:,3]
errrp = v[:,4]

# numerical error versus h
plt.figure(figsize=(7,6))
q1 = np.polyfit(np.log(h),np.log(err1),1)
q2 = np.polyfit(np.log(h),np.log(err2),1)
qrp = np.polyfit(np.log(h),np.log(errrp),1)
print('|s-sexact|_1 = O(h^%.2f), |s-sexact|_2 = O(h^%.2f), |H^r - Hexact^r|_p = O(h^%.2f)' \
      % (q1[0],q2[0],qrp[0]))
plt.loglog(h,err1,
           'k'+MARKER[0],ms=MARKERSIZE[0],mfc=MARKERFACE[0],
           label=NAME[0] + '$=O(h^{%.2f})$' % q1[0])
plt.loglog(h,np.exp(q1[0]*np.log(h)+q1[1]),'k--')
plt.loglog(h,err2,
           'k'+MARKER[1],ms=MARKERSIZE[1],mfc=MARKERFACE[1],
           label=NAME[1] + '$=O(h^{%.2f})$' % q2[0])
plt.loglog(h,np.exp(q2[0]*np.log(h)+q2[1]),'k--')
plt.loglog(h,errrp,
           'k'+MARKER[2],ms=MARKERSIZE[2],mfc=MARKERFACE[2],
           label=NAME[2] + '$=O(h^{%.2f})$' % qrp[0])
plt.loglog(h,np.exp(qrp[0]*np.log(h)+qrp[1]),'k--')
plt.grid(True)
plt.axis([1.0e-2,1.0e3,1.0e2,1.0e12])
plt.xlabel('$h$  (km)',fontsize=18.0)
plt.ylabel('numerical error',fontsize=18.0)
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.legend(loc='upper left',fontsize=14.0)
writeout('convergence.pdf')
