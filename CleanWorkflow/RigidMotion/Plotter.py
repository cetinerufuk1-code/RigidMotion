import csv
import numpy as np
import pandas as pd 
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

df = pd.read_csv(
    "FxData_NewRefRigidMotion_KC_0.001_m_3.0.csv",
    header=None
)
data = df.to_numpy()


# Recorded Data
t = data[:,0]
FxSigma = data[:,1]
FxRes = data[:,2]
Hussey = data[:,3]
vel = data[:,4]

# Run Parameters
amp = data[0,5]
diam = data[0,6]
Re = data[0,7]
KC = data[0,8]
m = data[0,9]
W = data[0,10]

mask = (t >= 0) & (t <= 3)

# Error Calculation (absolute norm)
HusseyMax = np.max(np.abs(Hussey[mask]))
MaxDiff = np.max(np.abs(FxRes[mask]-Hussey[mask]))
normErr = MaxDiff/HusseyMax


## Plotting
fig, ax1 = plt.subplots() # subplots ve twinx

ax1.plot(t[mask],Hussey[mask],color=colors[2], label="Hussey/Vujacic Reference",linewidth=3,alpha=0.5)
ax1.plot(t[mask],FxRes[mask],color=colors[0],label="Numerical Data",linestyle="-.")



ax2 = ax1.twinx()
ax2.plot(t[mask],vel[mask]/np.max(abs(vel)),color=colors[4],linestyle="--",label = "Velocity")
ax2.set_ylabel('Normalized Velocity, V/Vmax', color=colors[4])
ax2.tick_params(axis='y', labelcolor=colors[4])


ax1.set_title(f"KC = {KC:.3f} m = {m:.2f}, Re = {Re:.3f}, Max Norm Err: {normErr:.4f}")
ax1.set_ylabel('Normalized Inline Force')
ax1.set_xlabel('Phase Position, t/T')

ax1.legend()
ax2.legend()
plt.show()



