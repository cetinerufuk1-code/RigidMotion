import csv
import numpy as np
from math import *
import matplotlib.pyplot as plt
colors = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#E45756",  # red
    "#72B7B2",  # teal
    "#B279A2",  # purple
]
# =========
# Data
# ==========

data = np.loadtxt("RigidMotionHighRe.csv",delimiter=",")

# Recorded Data
t = data[:,0]
FxNum = data[:,1]
Hussey = data[:,2]
vel = data[:,3]

# Run Parameters
amp = data[1,4]
diam = data[1,5]
Re = data[1,6]
KC = data[1,7]
m = data[1,8]
W = data[1,9]

mask = (t >= 0) & (t <= 3)

# Error Calculation (absolute norm)
HusseyMax = np.max(np.abs(Hussey[mask]))
MaxDiff = np.max(np.abs(FxNum[mask]-Hussey[mask]))
normErr = MaxDiff/HusseyMax


## Plotting
fig, ax1 = plt.subplots() # subplots ve twinx

ax1.plot(t[mask],FxNum[mask],color=colors[0],label="Numerical Data")
ax1.plot(t[mask],Hussey[mask],color=colors[2], label="Hussey/Vujacic Reference")


ax2 = ax1.twinx()
ax2.plot(t[mask],vel[mask]/np.max(abs(vel)),color=colors[4],linestyle="--",label = "Velocity")
ax2.set_ylabel('Normalized Velocity, V/Vmax', color=colors[4])
ax2.tick_params(axis='y', labelcolor=colors[4])


ax1.set_title(f"KC = {KC:.2f} m = {m:.2f}, Re = {Re:.3f}, Max Norm Err: {normErr:.4f}")
ax1.set_ylabel('Normalized Inline Force')
ax1.set_xlabel('Phase Position, t/T')

ax1.legend()
ax2.legend()
plt.show()



