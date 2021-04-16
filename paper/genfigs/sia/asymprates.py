#!/usr/bin/env python3

# see py/1D/study/siaasymprates.sh  to generate input data  asymprates.txt

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

INFILE = 'asymprates.txt'
MARKER = ['o','s','D','+','+','+']
MARKERSIZE = [10.0,10.0,9.0,13.0,10.0,7.0]
NAME = ['V(0,2)', 'V(1,2)', 'V(2,2)',
        'V(0,2)-Jacobi0.6', 'V(0,2)-Jacobi0.5', 'V(0,2)-Jacobi0.4']

print('reading %s ...' % INFILE)
v = np.array(np.loadtxt(INFILE))
#print(v)

N = 4*6
assert len(v[:,0]) == N

# columns:  solver J m rate
solver = v[:,0]
L = 1800.0e3  # m
m = v[:,2]
h = L / (m+1)
h /= 1000.0
rate = v[:,3]

# numerical error versus h
plt.figure(figsize=(7,6))
for k in range(6):
    plt.semilogx(h[solver==k], rate[solver==k],
                 'k'+MARKER[k], ms=MARKERSIZE[k], mfc='w', label=NAME[k])
plt.grid(True)
plt.axis([0.3,100.0,0.5,1.0])
plt.xlabel('$h$  (km)',fontsize=18.0)
plt.ylabel('asymptotic rate',fontsize=18.0)
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.legend(loc='lower left',fontsize=14.0)
writeout('asymprates.pdf')
