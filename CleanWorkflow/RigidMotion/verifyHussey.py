# Forced Motion Oscullating Cylinder Verification Data
#
#
#

import numpy as np
from math import *
import matplotlib.pyplot as plt

## # R. G. Hussey and Peter Vujacic Data
## Table I, Undapmed  Coefficients for
## Circular Cylinder

m = [
    0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
    1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,
    2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,
    3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0
]

k = [
    19.6995,9.1655,6.1660,4.7708,3.9680,3.4472,3.0822,2.8122,2.6044,2.4395,
    2.3054,2.1943,2.1007,2.0207,1.9516,1.8913,1.8382,1.7910,1.7490,1.7111,
    1.6770,1.6459,1.6176,1.5917,1.5679,1.5459,1.5255,1.5067,1.4891,1.4727,
    1.4574,1.4430,1.4296,1.4169,1.4049,1.3936,1.3829,1.3728,1.3632,1.3541
]

kp = [
    48.630,16.726,9.2584,6.1849,4.5666,3.5863,2.9363,2.4769,2.1368,1.8757,
    1.6695,1.5028,1.3655,1.2505,1.1529,1.0692,0.99649,0.93287,0.87674,0.82687,
    0.78227,0.74217,0.70591,0.67299,0.64296,0.61547,0.59020,0.56690,0.54535,0.52537,
    0.50678,0.48945,0.47326,0.45810,0.44387,0.43049,0.41789,0.40600,0.39476,0.38412
]

m , k , kp = np.array(m), np.array(k), np.array(kp)

# Linear Interpolation for m**2 * k0'
#  m**2 * (k0 - 1)
def InterpData(mQ):
    kInterp = m**2 * (k - 1)
    kPInterp = m**2 * kp

    kInt = np.interp(mQ, m, kInterp)
    kPInt = np.interp(mQ, m, kPInterp)

    # recover k0 and k0'
    k0 = kInt / (mQ**2) + 1
    k0P = kPInt / (mQ**2)

    return k0, k0P

def getReferenceData(t,amp,w,diam,rhof,mQ):
    # ====================
    #  Helper Functions
    # ====================
    def circVel(t,A=amp,w=w):
        return -A*w*cos(w*t)

    def  circAcc(t,A=amp,w=w):
        return A*w**2*sin(w*t)

    # ====================
    #  Validation Datas
    # ===================
    k, kp = InterpData(mQ)  # k, k'
    a = diam/2

    def FxCalcTheor1(t,a=a,rhof=rhof,k=k,kp=kp,w=w):
        acc, vel = circAcc(t), circVel(t) 
        return -(pi*a**2*rhof)*(k*acc+kp*vel*w)
    
    HusseyRef = []
    for i in t:
        HusseyRef.append(FxCalcTheor1(i))

    return HusseyRef













