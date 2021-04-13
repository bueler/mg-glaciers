#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "cm"
fsize=16.0
bigfsize=20.0

m = 6  # number of internal nodes
h = 1.0 / (m+1)
x = np.linspace(0.0,1.0,m+2)

# Poisson problem:  - u'' = f,  u(0) = u(1) = 0

def residual(f,w):
    F = np.zeros(m+2)
    F[1:-1] = (1.0/h) * (2.0 * w[1:-1] - w[0:-2] - w[2:]) - h * f[1:-1]
    return F

def gspoint(f,w,p):
    w[p] = (h**2 * f[p] + w[p-1] + w[p+1]) / 2.0

uex = np.sin(np.pi * x)
frhs = np.pi**2 * uex

# plot ell[v] as piecewise-constant at interior points
def dualplot(x,ell):
    for p in range(1,m+1):
        # for some reason +- h/2 does not align properly
        plt.plot([x[p]-h/2.3,x[p]+h/2.3],[ell[p],ell[p]],'k',lw=2.0)

# initial w with both high frequency and low-frequency errors
np.random.seed(1)   # fix the random seed for repeatability
w = np.random.randn(m+2) + 5.0*uex
w[0] = 0.0
w[-1] = 0.0

# figure
xdelta = 2.0 * h
ydelta = 1.7
plt.figure(figsize=(10,8))
F = residual(frhs,w)
ew = w - uex
rscale = 1.0 / (max(F) - min(F))
ewscale = 1.0 / (max(ew) - min(ew))
for p in range(m+1):
    # plot scaled residual and zeroed location
    xshift = p * xdelta
    yshift = - p * ydelta
    plt.plot(x+xshift,np.zeros(m+2)+yshift,'k:',lw=1.0)
    if p > 0:
        gspoint(frhs,w,p)
        F = residual(frhs,w)
        plt.plot(x[p]+xshift,F[p]+yshift,'ko',ms=10.0,mfc='w')
    dualplot(x+xshift,rscale*F+yshift)
    # plot error
    ew = w - uex
    plt.plot(x+xshift+1.7,np.zeros(m+2)+yshift,'k:',lw=1.0)
    plt.plot(x+xshift+1.7,ewscale*ew+yshift,'k',lw=2.0)

plt.text(0.5,-8.0,'residuals',rotation=-62.0,fontsize=fsize)
plt.text(3.6,-4.0,'errors',rotation=-62.0,fontsize=fsize)
plt.axis('off')
writeout('residualpoints.pdf')
