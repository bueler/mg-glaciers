#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

INFILE = 'results-converge.txt'
LABELS = ['NGS only', 'V(1,1) cycles']
SYMBOLS = ['o','.']
MFC = ['w','k']
MSIZE = [12.0,12.0]

SHOW = False

def writeout(outname):
    if SHOW:
        plt.show()
    else:
        print('writing file ' + outname)
        plt.savefig(outname,bbox_inches='tight')

print('reading ' + INFILE + ' ...')
v = np.reshape(np.array(np.loadtxt(INFILE)), (-1,3))
# columns of v are:   method, m, |u-u_ex|_2
#print(v)

method = v[:,0]
m = v[:,1]
h = 1.0 / m
err = v[:,2]

plt.figure(figsize=(7,6))
for k in range(2):
    plt.loglog(h[method==k],err[method==k],'k' + SYMBOLS[k],
               mfc=MFC[k],label=LABELS[k],ms=MSIZE[k])
p = np.polyfit(np.log(h[method==1]),np.log(err[method==1]),1)
plt.loglog(h[method==1],np.exp(p[0]*np.log(h[method==1])+p[1]),'k:',
           label='$O(h^{%.2f})$' % p[0])

ax = plt.gca()
ax.set_xlim([1.0e-5,1.0e-1])
ax.set_ylim([1.0e-9,1.0e-1])
#plt.xticks([1.0,0.1,0.01,0.001,0.0001,0.00001],fontsize=12.0)
plt.xlabel('$h$',fontsize=16.0)
plt.ylabel('$|u-u_{ex}|_2$',fontsize=16.0)
plt.legend(loc='best',fontsize=14.0)
plt.minorticks_off()
plt.grid(True)

writeout('converge.pdf')

