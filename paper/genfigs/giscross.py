#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from writeout import writeout

df = pd.read_csv('gis/gris_cross_profile_lat_70.csv')
x = np.array(df["Profile [m]"]) / 1000.0
b = np.array(df["Bed [m]"])
s0 = np.array(df["Surface [m]"])
s = s0.copy()
s[s0 < 10.0] = b[s0 < 10.0]
H = s - b
L = 750.0

# s(x) and b(x) versus x, with 100x exaggeration
plt.figure(figsize=(10,7))
plt.subplot(2,1,1)
plt.plot(x,s,'k',label='$s$')
plt.plot(x,b,'k--',label='$b$')
plt.grid(True)
plt.xlabel('')
plt.ylabel('elevation  (m)',fontsize=18.0)
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.axis([0.0,L,-1000.0,3500.0])
plt.legend(loc='upper left',fontsize=14.0)
fig = plt.subplot(2,1,2)
plt.plot(x,H,'k')
plt.grid(True)
plt.xlabel('$x$  (km)',fontsize=18.0)
plt.ylabel('$H$  (m)',fontsize=18.0)
plt.xticks(fontsize=14.0)
plt.yticks(fontsize=14.0)
plt.minorticks_off()
plt.axis([0.0,L,-100.0,3500.0])
im = plt.imread('gis/gris-profile-gray.png')  # image is 200 pixels tall
plt.figimage(im, 135.0, 100.0)  # offset in pixels
writeout('giscross.pdf')
