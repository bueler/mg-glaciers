#!/usr/bin/env python3

# see py/1D/study/plapconv.sh  to generate input data  plapconv.txt

# figure here not for inclusion in paper, but we do the same log-linear fit as in siaconv.py to compare rates

import numpy as np
import matplotlib.pyplot as plt

INFILE = 'convergence.txt'

print('reading %s ...' % INFILE)
v = np.array(np.loadtxt(INFILE))
#print(v)

N = 13  # number of different resolutions
assert len(v[:,0]) == N

# columns:  J m err2 errp
m = v[:,1]
h = 1.0 / (m+1)
errp = v[:,3]

q = np.polyfit(np.log(h),np.log(errp),1)
print('|u-uexact|_p = O(h^%.2f)' % q[0])

SHOW = False
def writeout(outname):
    if SHOW:
        plt.show()
    else:
        print('writing file ' + outname)
        plt.savefig(outname,bbox_inches='tight')

# numerical error versus h
plt.figure(figsize=(7,6))
plt.loglog(h,errp,
           'ko',ms=10.0,mfc='k',
           label='$|u-u_{exact}|_p=O(h^{%.2f})$' % q[0])
plt.loglog(h,np.exp(q[0]*np.log(h)+q[1]),'k--')
plt.grid(True)
plt.xlabel('$h$',fontsize=18.0)
plt.ylabel('numerical error',fontsize=18.0)
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.legend(loc='upper left',fontsize=14.0)
writeout('convergence.pdf')
