#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from writeout import writeout

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

fsize=15.0

def makeaxes(xmin,xmax,ymin,ymax):
    x = np.linspace(xmin,xmax,2)
    y = np.linspace(ymin,ymax,2)
    zero = np.zeros(np.shape(x))
    plt.plot(x,zero,'k',lw=1.0)
    plt.plot(zero,y,'k',lw=1.0)

# H, K, and contour lines figure
plt.figure(figsize=(8,6))
makeaxes(-0.25,1.2,-0.1,0.8)
phix = 0.1
phiy = 0.3
plt.fill([phix,1.2,1.2,phix,phix],[phiy,phiy,0.8,0.8,phiy],'k',alpha=0.2)
xhat = phix + 0.25
yhat = phiy
yc = phiy - 0.3
plt.plot(phix,phiy,'k.',ms=10.0)
plt.text(phix-0.05,phiy-0.08,r'$\varphi$',fontsize=fsize)
plt.plot(xhat,yhat,'k.',ms=10.0)
plt.text(xhat,yhat-0.08,r'$u$',fontsize=fsize)
plt.plot(xhat+0.2,yhat+0.2,'k.',ms=10.0)
plt.text(xhat+0.15,yhat+0.22,r'$v$',fontsize=fsize)
plt.gca().annotate('', xy=(xhat+0.2, yhat+0.2), xytext=(xhat, yhat),
                   arrowprops=dict(arrowstyle='->, head_length=0.8, head_width=0.5'))
plt.text(xhat+0.1,yhat+0.05,r'$v-u$',fontsize=fsize)
plt.gca().annotate('', xy=(xhat, yhat+0.38), xytext=(xhat, yhat),
                   arrowprops=dict(arrowstyle='->, head_length=0.8, head_width=0.5'))
plt.text(xhat-0.17,yhat+0.21,r'$F(u)$',fontsize=fsize)
plt.plot(xhat,yc+0.06,'ko',ms=8.0,mfc='w')
plt.text(xhat+0.05,yc+0.05,'$\it{unconstrained}\,\, \it{minimizer}$',fontsize=12.0)
ax = plt.gca()
for r in [0.4, 0.5, 0.6, 0.7]:
    alpha = (180.0/np.pi) * np.arcsin(0.3/r)
    beta = 90.0 + (180.0/np.pi) * np.arcsin(0.25/r)
    beta = min(beta,180.0 - alpha)
    arc = Arc((xhat,yc), 2.0*r, 2.0*r, angle=0.0, theta1=alpha, theta2=beta,
              color='k', lw=0.75)
    ax.add_patch(arc)
plt.text(-0.2,0.7,r'$\mathcal{H}$',fontsize=fsize)
plt.text(0.9,0.6,r'$\mathcal{K}_\varphi$',fontsize=fsize)
plt.axis('tight')
plt.axis('off')
plt.axis('equal')
writeout('cartoonplane.pdf')

# inner cone approx figure
plt.figure(figsize=(8,6))
xm, ym = 0.6, 0.4
makeaxes(-0.4, xm, -0.25, ym)
cornerx = [-0.3, -0.29, -0.2, -0.1]
cornery = [-0.2, -0.14, -0.09, 0.0]
sh = 0.02
for n, xc, yc in zip(range(4), cornerx, cornery):
    plt.fill([xc,xm,xm,xc,xc],[yc,yc,ym,ym,yc],'k',alpha=0.13)
    plt.plot(cornerx[n],cornery[n],'k.',ms=10.0)
    plt.text(cornerx[n]+(n-3)*sh,cornery[n]-2*sh,'$\mathcal{X}^%d$' % (3-n),
             fontsize=fsize)
plt.plot(0.0,0.0,'k.',ms=10.0)
plt.text(0.0+sh,0.0+sh,r'$0$',fontsize=fsize)
plt.text(-0.4,0.35,r'$\mathcal{V}^3$',
         fontsize=fsize)
plt.text(0.4,0.03,r'$\mathcal{D}^0=\mathcal{K}^0$',
         fontsize=fsize)
plt.text(0.33,cornery[2]+0.02,r'$\mathcal{D}^1=\mathcal{K}^0+\mathcal{K}^1$',
         fontsize=fsize)
plt.text(0.27,cornery[1]+0.01,r'$\mathcal{D}^2=\mathcal{K}^0+\mathcal{K}^1+\mathcal{K}^2$',
         fontsize=fsize)
plt.text(0.19,cornery[0]+0.01,
         r'$\mathcal{D}^3=\mathcal{K}^0+\mathcal{K}^1+\mathcal{K}^2+\mathcal{K}^3$',
         fontsize=fsize)
plt.axis('tight')
plt.axis('off')
plt.axis('equal')
writeout('innerconeapprox.pdf')
