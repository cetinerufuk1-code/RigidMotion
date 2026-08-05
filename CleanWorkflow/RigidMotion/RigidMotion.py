from ngsolve import *
from netgen.occ import *
import ipywidgets as widgets
import matplotlib.pyplot as plt
from netgen.meshing import MeshingParameters

import csv 
import numpy as np
import sys

from verifyHussey import getReferenceData # Generate Reference Data for the Current Set

# ======================
#   Parameters 
# ======================

# Set Parameters
no_osc = 5 # no of oscillations to be simulatied
T_dt = 360 # no of timesteps per oscillation
order = 2

rhof, nuf, U = 1e3, 1e-6, 0 # Fluid Properties
mu = rhof * nuf

diam = 0.01 # !!
KC, m = float(sys.argv[1]), float(sys.argv[2]) # !!

### Derived Quantities
amp = (KC * diam) / (2*pi)
W = ((4 * m) / diam)**2 * nuf
f = W / (2*pi)
tend = no_osc / f
tau = 1 / (T_dt* f)

RFx = rhof * pi * (0.5*diam)**2 * W**2 * amp     # Reference Force Per Length ϱπ𝑎²ω𝑐
Re = 2*pi*f*amp * diam / nuf

# Data Output (Change Name)
fileNameCsv = "FxData_NewRefRigidMotion_KC_" + str(KC) + "_m_" + str(m) + ".csv"

# Debug Lines
print("Reynolds Number: " + str(Re) )
print("KC Number: " + str((2*pi*amp)/diam))
print("m:" + str(0.25*diam*sqrt(W/nuf)))
print("Frequency: " + str(f))
print("Amplitude-Diameter Ratio: " + str(amp/diam))

def GenerateMesh(order, h=diam*3.5,grading=0.1):
    circle = Circle((0,0),r= diam/2).Face()
    circle.edges.name = "circ"

    fluid = Circle((0,0),r=diam*150).Face()
    fluid.edges.name = "outlet"
    fluid.faces.name = "fluid"

    domain_fluid = fluid - circle
    domain_fluid.edges[1].maxh = (diam * 0.01)

    mp = MeshingParameters(maxh=h,grading=grading)
    R = diam * 7.5
    hfine = diam * 0.5

    # Local Mesh Refinement
    for r in np.linspace(0, R, 20):
        for theta in np.linspace(0, 2*np.pi, 40):
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            mp.RestrictH(x=x, y=y, z=0, h=hfine)
        
    mesh = Mesh(OCCGeometry(domain_fluid,dim=2).GenerateMesh(mp))
    mesh.Curve(order+1)

    return mesh

mesh = GenerateMesh(order=order+1)
print("Mesh Cell Count:" + str(mesh.ne))

## Rigid Grid Motion
# Define Cylinder Motion
def circDisp(t,A=amp,w=W):
    return -A*sin(w*t) 

def circVel(t,A=amp,w=W):
    return -A*w*cos(w*t)

## Grid for rigid mesh motion
D = VectorH1(mesh, order=order)
gfd, gfd_old = GridFunction(D), GridFunction(D)

# Grid Deformation Quantities
I = Id(mesh.dim)
def GridDef(A):
    F = A + I
    J = Det(F)
    C = F.trans * F
    Finv = Inv(F)
    return (F, J, C, Finv)

F, J, C, Finv = GridDef(Grad(gfd))
F_old, J_old, C_old , Finv_old = GridDef(Grad(gfd_old))
gradd_old = Grad(gfd_old)

## Define Fluid Space
V = VectorH1(mesh,order=order, dirichlet = "circ")
Q = H1(mesh, order = order - 1, definedon="fluid")

X = V * Q
(u,p) , (v,q) = X.TnT()

gf_solution, gf_solution_old = GridFunction(X), GridFunction(X)
velocity, pressure = gf_solution.components
velocity_old, pressure_old = gf_solution_old.components

gradu_old = Grad(velocity_old)

## FLUID SOLVE STEP
true_compile = False
bfa = BilinearForm(X, symmetric=False)

# M du/dt
bfa += (rhof / tau * (InnerProduct(0.5 * (J + J_old) * (u - velocity_old), v))).Compile(
    true_compile, wait=True
) * dx("fluid")

# symmetric stress div (eps u)
bfa += (
    0.5
    * rhof
    * nuf
    * (
        InnerProduct(J * 2 * Sym(Grad(u) * Finv), Sym(Grad(v) * Finv))
        + InnerProduct(J_old * 2 * Sym(gradu_old * Finv_old), (Grad(v) * Finv_old))
    )
).Compile(true_compile, wait=True) * dx("fluid")

# Convection and mesh-velocity
bfa += (
    0.5
    * rhof
    * (
        InnerProduct(J * (Grad(u) * Finv) * (u - (gfd - gfd_old) / tau), v)
        + InnerProduct(
            J_old
            * (gradu_old * Finv_old)
            * (velocity_old - (gfd - gfd_old) / tau),
            v,
        )
    )
).Compile(true_compile, wait=True) * dx("fluid")

# Pressure/Constraint implicit
bfa += (-J * (Trace(Grad(v) * Finv) * p + Trace(Grad(u) * Finv) * q)).Compile(
    true_compile, wait=True
) * dx("fluid")

# ===== Helper Functions
# X-Force Calculator (verify formulation)
def FxCalc(I=I,F=F,J=J,Finv=Finv,mu=mu):
    sigma_ale = -pressure * I + 2 * mu * Sym(Grad(velocity) * Finv)
    nref = specialcf.normal(mesh.dim)
    Fx = Integrate((J * sigma_ale * Finv.trans * nref)[0],
               mesh.Boundaries("circ"),order=10)
    return Fx

# ================================
#    Time Loop
# ===============================

pos, tHistory, Fx = [], [], []
t = -(1/f)*0.25 # Start one quarter of a cycle early so that velocity is at min
i = 0

# VTK File for Visualization
# fileNameVTK = "C:\\Users\\cetin\\MyLocalFiles\\UROP2\\RigidMotion\\VTKOut\\RigidMotion" + getDate.strftime("%H-%M-%S") 
#vtkU = VTKOutput(mesh,coefs=[velocity,pressure],names=["Velocity","Pressure"],filename=fileNameVTK,subdivision=2)
#vtkU.Do(time = t)

firstStep = True

with TaskManager():
    with open(fileNameCsv, "w", newline="") as c:
        writer = csv.writer(c)

        while t < tend - tau / 2 : 
            # Update Solutions
            gf_solution_old.vec.data = gf_solution.vec
            gfd_old.vec.data = gfd.vec
            t += tau

            # Prescribe Rigid Displacement
            gfd.Set(CF((circDisp(t),0)))
            velocity.Set(CF((circVel(t),0)))

            # Solve for Fluid 
            solvers.Newton(bfa,gf_solution,maxit = 50, maxerr = 1e-12, printing=False)
            
            # Data Recording (in case of a crash)
            tHistory.append(t) # t/T
            fx = FxCalc()
            Fx.append(fx)

            # Get Reference Data  
            HusseyRef= getReferenceData(tHistory,amp,W,diam,rhof,m)
            # Record Data
            if firstStep:
                writer.writerow([t*f,fx/RFx,HusseyRef[-1]/RFx,circVel(t),amp,diam,Re,KC,m,W])
                firstStep = False
            else:
                writer.writerow([t*f,fx/RFx,HusseyRef[-1]/RFx,circVel(t)])

            # Error Calculation
            if t*f > 0:
                mask = t*f > 0
                HusseyMax = np.max(np.abs(HusseyRef))
                DiffCurr =np.abs(Fx[-1]-HusseyRef[-1])
                normErr = DiffCurr/HusseyMax
                print("Current Norm Err:" + str(normErr))

            print("  |     t/T         | |  In-Line Force(N/m)   | |    Hussey(N/m)   ")
            print(f" |  {t*f:.8f}      | |    {fx/RFx:.8f}       | |   {HusseyRef[-1]/RFx:.8f}    ")

            if i % 25 == 0:
                c.flush()

            i += 1