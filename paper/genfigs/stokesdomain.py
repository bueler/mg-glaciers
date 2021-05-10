#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

fsize=18.0
smallfsize=14.0
extrasmallfsize=12.0
bigfsize=20.0

def genbasicfig():
    x = np.linspace(0.0,10.0,1001)
    # bed elevation
    b = 0.07*(x-3.0)**2 + 0.2*np.sin(2.0*x) - 0.1
    plt.plot(x, b, 'k', lw=2.5)
    # current thickness for Omega^{n-1}
    h0 = 3.0
    L = 3.0
    firstshape = h0*(-0.2 + np.sqrt(np.maximum(0.0,1.0 - (x-5)**2/L**2)))
    thk = np.maximum(0.0, firstshape)
    # surface
    s = b + thk
    offset = 0.1
    plt.plot(x, s + offset, 'k--', lw=3.0)
    # reset axes
    plt.axis([0.0-0.02,10.0+0.02,-0.5,4.5])
    plt.axis('off')
    return x, s, b

def drawclimate(x,s,mycolor):
    plt.text(x[10], s[10]+2.7, r'$a(x,y)$', fontsize=bigfsize, color='k')
    for j in range(10):
        xarr = x[50+100*j]
        if j>0:
            magarr = 0.6*np.sin(np.pi/2 + 0.6*xarr)
        else:
            magarr = 0.05
        plt.arrow(xarr, s.max()+0.3, 0.0, magarr,
                  lw=1.5, head_width=0.1, color=mycolor)

# domain notation figure
plt.figure(figsize=(10,5.5))
x, s, b = genbasicfig()
plt.text(x[600] - 1.0, b[600] + 0.4 * s[600], r'$\Lambda_s$',
         fontsize=bigfsize, color='k')
drawclimate(x,s,'k')
# mark top surface
plt.text(x[250], s[250] + 1.5, r'$s(x,y)$', fontsize=bigfsize, color='k')
#plt.annotate(r'$\overline{\partial}\Lambda$',fontsize=fsize,
#             xy=(x[300],s[300]),
#             xytext=(x[300]-1.5,s[300]+0.5),
#             arrowprops=dict(facecolor='black', width=0.5, headwidth=5.0, shrink=0.1))
# mark bottom surface
plt.text(x[650], b[650] - 0.5, r'$b(x,y)$', fontsize=bigfsize, color='k')
# BIZARRE HACK NEEDED BECAUSE \underline{} THROWS ERROR
#plt.annotate(r"$\underline{\partial}\Lambda$",fontsize=fsize,
#plt.annotate(r"$\partial\Lambda$",fontsize=fsize,
#             xy=(x[700],b[700]),
#             xytext=(x[700]+1.1,b[700]-1.0),
#             arrowprops=dict(facecolor='black', width=0.5, headwidth=5.0, shrink=0.1))
#plt.text(x[700]+1.08,b[700]-1.02,r"$\_$",fontsize=24.0)  # HACK UNDERLINE
# show \pi\Lambda_s
#ypi = min(b) - 0.5
#plt.plot([min(x[s>b]), max(x[s>b])], [ypi,ypi],
#         color='k', lw=1.0)
#plt.text(3.5, ypi-0.5, r'$\pi\Lambda_s$', fontsize=bigfsize)
# show \Omega
yR = min(b) - 0.5
plt.plot([min(x),max(x)],[yR,yR],color='k',lw=1.0)
plt.text(2.0,yR-0.4,r'$\Omega$',fontsize=fsize)
plt.axis([0.0,10.0,yR-0.8,4.5])
writeout('stokesdomain.pdf')
