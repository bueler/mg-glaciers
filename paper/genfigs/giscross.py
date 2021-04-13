#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

SHOW = False
def writeout(outname):
    if SHOW:
        plt.show()
    else:
        print('writing file ' + outname)
        plt.savefig(outname,bbox_inches='tight')

INFILES = ['gis/topg_j1100_69Nlat.txt', 'gis/thk_j1100_69Nlat.txt']
f = INFILES[0]
print('reading b(x) from %s ...' % f, end='')
v = np.array(np.loadtxt(f))
x1 = v[:,0]
b = v[:,1]
print(' %d values read' % len(x1))
f = INFILES[1]
print('reading H(x) from %s ...' % f, end='')
v = np.array(np.loadtxt(f))
x2 = v[:,0]
H = v[:,1]
print(' %d values read' % len(x2))
assert max(abs(x1-x2)) == 0
#sys.exit(0)

# show on interval [0,1200km] starting 200km in from left
x = -250.0 + (x1 - min(x1)) / 1000.0
L = 1000.0
s = b + H

# s(x) and b(x) versus x, with 100x exaggeration
plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(x,s,'k--',label='$s$')
plt.plot(x,b,'k',label='$b$')
plt.grid(True)
plt.xlabel('')
plt.ylabel('elevation  (m)',fontsize=18.0)
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.axis([0.0,L,-1000.0,3200.0])
plt.legend(loc='upper left',fontsize=14.0)
plt.subplot(2,1,2)
plt.plot(x,H,'k')
plt.grid(True)
plt.xlabel('$x$  (km)',fontsize=18.0)
plt.ylabel('$H$  (m)',fontsize=18.0)
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.axis([0.0,L,-100.0,3200.0])
writeout('giscross.pdf')
