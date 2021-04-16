#!/usr/bin/env python3

# see py/1D/study/perfni.sh  to generate input data perfni.txt

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

INFILE = 'perfni.txt'
SOLVERNAME = ['discretization error', 'F-cycle', 'F-cycle (2xV)']
MARKER = ['*','o','s']
MARKERSIZE = [8.0,10.0,10.0]
MARKERFACE = ['k','w','w']

print('reading %s ...' % INFILE)
v = np.array(np.loadtxt(INFILE))

N = 11  # number of different resolutions
assert len(v[:,0]) == 3*N

# columns:  solver J m WU error
solver = v[:,0]
m = v[:,2]
h = 1.0 / (m+1)
WU = v[:,3]
err = v[:,4]

plt.figure(figsize=(7,6))
for s in [1, 2, 0]:  # s=solver
    mm = m[solver==s]
    eerr = err[solver==s]
    plt.loglog(mm,eerr,'k'+MARKER[s],ms=MARKERSIZE[s],mfc=MARKERFACE[s],
               label=SOLVERNAME[s])
plt.grid(True)
plt.xlabel('$m$',fontsize=18.0)
plt.ylabel('error norm',fontsize=18.0)
#plt.axis([100.0,1.0e5,0.0,55.0])
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.legend(loc='lower left',fontsize=14.0)
writeout('perfni.pdf')
