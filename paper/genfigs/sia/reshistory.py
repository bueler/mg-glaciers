#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

INFILE = 'reshistory.txt'

print('reading %s ...' % INFILE)
res = np.array(np.loadtxt(INFILE, usecols=3))
N = len(res)
assert N == 2*31

sweeps = res[:31]
vcycles = res[31:]

# numerical error versus h
plt.figure(figsize=(7,6))
plt.semilogy(range(31), sweeps, 'k.', ms=12.0, mfc='w', label='single level')
plt.semilogy(range(31), vcycles, 'k.', ms=12.0, label='V(0,2)')
plt.grid(True)
plt.axis([-1.0,31.0,1.0e-3,1.0e4])
plt.xlabel('iteration',fontsize=18.0)
plt.ylabel('residual norm',fontsize=18.0)
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.legend(loc='upper right',fontsize=14.0)
writeout('reshistory.pdf')

