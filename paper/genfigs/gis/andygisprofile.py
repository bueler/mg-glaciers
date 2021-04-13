import pandas as pd
import pylab as plt
df = pd.read_csv("gris_cross_profile.csv")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(df["Profile [m]"], df["Bed [m]"], label="Bed")
ax.plot(df["Profile [m]"], df["Surface [m]"], label="Surface")
ax.legend()
plt.show()
