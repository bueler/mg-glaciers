#!/usr/bin/env python3
# (C) 2021 Ed Bueler

import numpy as np

profiledefaults = {'R' : 10000.0,
                   'H' : 1000.0}

def profile(mx, R=None, H=None):
    '''Exact solution (Bueler profile) with half-length (radius) R and
    maximum height H, on the interval [0,L] = [0,2R], centered at x=R.
    See van der Veen (2013) equation (5.50).  Generates numpy arrays x,s
    where x has mx+1 equally-spaced points on [0,L].'''
    n = 3.0                       # glen exponent
    p1 = n / (2.0 * n + 2.0)      # e.g. 3/8
    q1 = 1.0 + 1.0 / n            #      4/3
    Z = H / (n - 1.0)**p1         # outer constant
    x = np.linspace(0.0, 2.0 * R, mx+1)
    X = (x - R) / R               # rescaled coord
    Xin = abs(X[abs(X) < 1.0])    # rescaled distance from center
    Yin = 1.0 - Xin
    s = np.zeros(np.shape(x))     # correct outside ice
    s[abs(X) < 1.0] = Z * ( (n + 1.0) * Xin - 1.0 \
                            + n * Yin**q1 - n * Xin**q1 )**p1
    return x, s
