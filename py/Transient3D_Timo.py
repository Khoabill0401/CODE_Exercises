"""
#=========================================================================#
# This code was programmed by:                                            #
# 1. Nguyen Anh Khoa - 1810240                                            #
# 2. Trang Si Tan Khang - 1810215                                         #
# 3. Phan Vuong Phu - 1710235                                             #
# Transient analysis for space frame structure                            #
# Cross section area = 400x400mm = 160000mm2                              #
# Length of each beam = 6000mm = 6m                                       #
# Length of each column = 3500mm = 3.5m                                   #
# Young's Elastic Modulus = E = 32.5x10^3 MPa = 32500 N/mm2               #
# Shear Modulus of Elasticity = G = 13x10^3 MPa = 13000 N/mm2             #
#=========================================================================#
#                                                                         #
#                                      6m                                 #
#                            |--------------------|                       #
#                       (9)  H====================H ---  <---10kN         #
#                            H       [11]    (10) H  |                    #
#                            H                    H  |                    #
#                            H                    H  |                    #
#                        [9] H               [10] H  |                    #
#                            H                    H  |                    #
#                            H        [8]    (8)  H  |  2x3.5=7m          #
#               10kN--->(7)  H====================H  |  ---               #
#                         // H                 // H  |  /                 #
#                       //   H           [6] //   H   / 6m                #
#                 [5] // [3] H             // [4] H /|                    #
#                   //       H      (6)  //       /  |                    #
#     10kN--->(5)  H=====================H --- ---H  |                    #
#                  H     (3) H [7]       H  | (4) H ---                   #
#                  H                     H  |                             #
#                  H                     H  |3.5m             Z           #
#              [1] H                 [2] H  |                 |      Y    #
#                  H                     H  |                 |    /      #
#                  H                     H ---                |  /        #
#             (1)  |---------------------|  (2)               |/_______ X #
#                             6m                                          #
#                                                                         #
#=========================================================================#
"""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mydef.SpaceFrameElementStiffness_Timoshenko import *
from mydef.SpaceFrameElementMass import *
from mydef.SpaceFrameAssemble import *
from scipy.linalg import eigh
from mydef.Newmark import *

nel = 11                       # number of elements
nnel = 2                       # number of nodes each element
ndof = 6                       # number of degrees of freedom per node
nnode = 10                     # total number of nodes
sdof = nnode*ndof              # total number of degrees of freedom

# Young's elastic modulus (N/m^2)
el = 3.25e10

# Shear modulus of elasticity (N/m^2)
G = 16250000000 #1.3e10

# Shear correction factor
ks = 1

# Cross-sectional area (m^2)
area = 0.16

# Moment of inertia (m^4)
xi = 0.0256/12
yi = 0.0256/12
zi = 0.0256/12

# Polar moment of inertia (m^4)
J = 3.605e-3 #0

# Material density (N/m^3)
rho = 25000

# Nodal coordinate, in [x1,  x2,  x3,  x4,  x5,  x6,  x7,  x8,  x9,  x10]
#                      [y1,  y2,  y3,  y4,  y5,  y6,  y7,  y8,  y9,  y10]
#                      [z1,  z2,  z3,  z4,  z5,  z6,  z7,  z8,  z9,  z10]
x = np.array([0,   6,   0,   6,   0,   6,   0,   6,   0,   6], dtype=float)
y = np.array([0,   0,   6,   6,   0,   0,   6,   6,   6,   6], dtype=float)
z = np.array([0,   0,   0,   0, 3.5, 3.5, 3.5, 3.5,   7,   7], dtype=float)
# Element connection, 1:1-5; 2:2-6; 3:3-7; 4:4-8; 5:5-7; 6:6-8; 7:5-6; 8:7-8; 9:7-9; 10:8-10; 11:9-10
element = np.array(([[1, 2, 3, 4, 5, 6, 5, 7, 7, 8, 9],[5, 6, 7, 8, 7, 8, 6, 8, 9, 10, 10]]), dtype = int) -1

# Apply boundary condition
bcdof = np.zeros((sdof), dtype=int)
bcval = np.zeros((sdof), dtype=int)
# Node 1:
bcdof[0] = 0
bcval[0] = 0
bcdof[1] = 1
bcval[1] = 0
bcdof[2] = 2
bcval[2] = 0
bcdof[3] = 3
bcval[3] = 0
bcdof[4] = 4
bcval[4] = 0
bcdof[5] = 5
bcval[5] = 0
# Node 2:
bcdof[6] = 6
bcval[6] = 0
bcdof[7] = 7
bcval[7] = 0
bcdof[8] = 8
bcval[8] = 0
bcdof[9] = 9
bcval[9] = 0
bcdof[10] = 10
bcval[10] = 0
bcdof[11] = 11
bcval[11] = 0
# Node 3:
bcdof[12] = 12
bcval[12] = 0
bcdof[13] = 13
bcval[13] = 0
bcdof[14] = 14
bcval[14] = 0
bcdof[15] = 15
bcval[15] = 0
bcdof[16] = 16
bcval[16] = 0
bcdof[17] = 17
bcval[17] = 0
# Node 4:
bcdof[18] = 18
bcval[18] = 0
bcdof[19] = 19
bcval[19] = 0
bcdof[20] = 20
bcval[20] = 0
bcdof[21] = 21
bcval[21] = 0
bcdof[22] = 22
bcval[22] = 0
bcdof[23] = 23
bcval[23] = 0

# Force matrix
ff = np.zeros((sdof, 1), dtype=float)
# Stiffness matrix
kk = np.zeros((sdof, sdof), dtype=float)
# Horizontal force at node 5, 7 and 10 with value of 10kN
ff[24] =  10000 # N
ff[36] =  10000 # N
ff[54] = -10000 # N

# Stiffness matrix initiation
k0 = np.zeros((12, 12), dtype = float)
k1 = np.zeros((12, 12), dtype = float)
k2 = np.zeros((12, 12), dtype = float)
k3 = np.zeros((12, 12), dtype = float)
k4 = np.zeros((12, 12), dtype = float)
k5 = np.zeros((12, 12), dtype = float)
k6 = np.zeros((12, 12), dtype = float)
k7 = np.zeros((12, 12), dtype = float)
k8 = np.zeros((12, 12), dtype = float)
k9 = np.zeros((12, 12), dtype = float)
k10 = np.zeros((12, 12), dtype = float)
kk = np.zeros((sdof, sdof), dtype = float)

# Mass matrix initiation
m0 = np.zeros((12, 12), dtype = float)
m1 = np.zeros((12, 12), dtype = float)
m2 = np.zeros((12, 12), dtype = float)
m3 = np.zeros((12, 12), dtype = float)
m4 = np.zeros((12, 12), dtype = float)
m5 = np.zeros((12, 12), dtype = float)
m6 = np.zeros((12, 12), dtype = float)
m7 = np.zeros((12, 12), dtype = float)
m8 = np.zeros((12, 12), dtype = float)
m9 = np.zeros((12, 12), dtype = float)
m10 = np.zeros((12, 12), dtype = float)
mm = np.zeros((sdof, sdof), dtype = float)

for iel in range(nel):
    # Coordinate of node i
    x_i = x[element[0, iel]]
    y_i = y[element[0, iel]]
    z_i = z[element[0, iel]]
    # Coordinate of node j
    x_j = x[element[1, iel]]
    y_j = y[element[1, iel]]
    z_j = z[element[1, iel]]
    if iel == 0:
        k0 = SpaceFrameElementStiffness_Timoshenko(el, G, ks, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        m0 = SpaceFrameElementMass(area, rho, xi, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k0
        m = m0
    if iel == 1:
        k1 = SpaceFrameElementStiffness_Timoshenko(el, G, ks, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        m1 = SpaceFrameElementMass(area, rho, xi, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k1
        m = m1
    if iel == 2:
        k2 = SpaceFrameElementStiffness_Timoshenko(el, G, ks, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        m2 = SpaceFrameElementMass(area, rho, xi, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k2
        m = m2
    if iel == 3:
        k3 = SpaceFrameElementStiffness_Timoshenko(el, G, ks, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        m3 = SpaceFrameElementMass(area, rho, xi, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k3
        m = m3
    if iel == 4:
        k4 = SpaceFrameElementStiffness_Timoshenko(el, G, ks, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        m4 = SpaceFrameElementMass(area, rho, xi, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k4
        m = m4
    if iel == 5:
        k5 = SpaceFrameElementStiffness_Timoshenko(el, G, ks, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        m5 = SpaceFrameElementMass(area, rho, xi, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k5
        m = m5
    if iel == 6:
        k6 = SpaceFrameElementStiffness_Timoshenko(el, G, ks, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        m6 = SpaceFrameElementMass(area, rho, xi, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k6
        m = m6
    if iel == 7:
        k7 = SpaceFrameElementStiffness_Timoshenko(el, G, ks, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        m7 = SpaceFrameElementMass(area, rho, xi, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k7
        m = m7
    if iel == 8:
        k8 = SpaceFrameElementStiffness_Timoshenko(el, G, ks, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        m8 = SpaceFrameElementMass(area, rho, xi, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k8
        m = m8
    if iel == 9:
        k9 = SpaceFrameElementStiffness_Timoshenko(el, G, ks, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        m9 = SpaceFrameElementMass(area, rho, xi, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k9
        m = m9
    if iel == 10:
        k10 = SpaceFrameElementStiffness_Timoshenko(el, G, ks, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        m10 = SpaceFrameElementMass(area, rho, xi, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k10
        m = m10
    kk = SpaceFrameAssemble(kk, k, element[0, iel], element[1, iel])
    mm = SpaceFrameAssemble(mm, m, element[0, iel], element[1, iel])

# apply boundary
bcd = np.array(([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]), dtype = int)

# remove the fixed degrees of freedom
free_dof = np.setdiff1d(np.arange(0, sdof), bcd)
# Solve eigenvalue problem
(eigen_evalues, eigen_evectors) = eigh(kk[np.ix_(free_dof, free_dof)], mm[np.ix_(free_dof, free_dof)])
frequencies = np.sqrt(eigen_evalues)/2/math.pi
#print(frequencies)

Omega = math.sqrt(eigen_evalues[0])

full_eigen_evectors = np.zeros((sdof, eigen_evectors.shape[1]), dtype = float)
full_eigen_evectors[free_dof, :] = eigen_evectors


"""
#=========================================================================#
# TRANSIENT ANALYSIS USING THE NEWMARK METHOD                             #
#=========================================================================#
"""

# Define the dof of a node to display result
dof_output = 24
measuring_dofs = np.array([24, 30, 36, 42, 48, 54], dtype=int)

# Set up the time step and the relative parameters to calculate according to this method
deltaT = 2.50e-4                       # The time step
LengthTime = 2.50e-4*1e4 #2.5                       # The length of time for calculating
ns = int(LengthTime/deltaT+1)          # Number of time steps
Time = np.linspace(0, LengthTime, ns)  # Initialize the time values w.r.t applied load Fs
# Initialize the applied load vector (sdof x ns) w.r.t all the time steps
Fs = np.zeros((sdof, ns), dtype=float)
for s in range(ns):
    """
    # Step loading
    if Time[s] <= 0.1:
        Fs[np.arange(0, sdof).reshape(-1, 1), s] = 1 * F
    else:
        Fs[np.arange(0, sdof).reshape(-1, 1), s] = 0 * F
    #"""

    """
    # Triangular loading
    if Time[s] <= 0.1:
        Fs[np.arange(0, sdof).reshape(-1, 1), s] = (1-Time[s]/10e-3)*F
    else:
        Fs[np.arange(0, sdof).reshape(-1, 1), s] = 0*F
    #"""

    #"""
    # Sine loading
    Fs[np.arange(0, sdof).reshape(-1, 1), s] = ff*math.sin(10*Time[s])
    # """

    """
    # Explosive blast loading
    Fs[np.arange(0, sdof).reshape(-1, 1), s] = F*math.exp(-10*Time[s])
    #"""
# The deflection at dof dof_output versus the time t
(deflection, disp_res_vector, acce_res_vector) = Newmark(kk, mm, Fs, sdof, free_dof, dof_output, deltaT, ns, measuring_dofs)

# Plot the deflection history versus the time t
fig = plt.figure(99)
FS = 12
plt.plot(Time, deflection.flatten(), 'b--', linewidth=1.0)
plt.title('Deflection versus time', fontname="Segoe UI", fontsize=FS + 2, color='red')
# plt.xlabel('Number of PF calls', fontname = "Segoe UI", fontsize=FS, color='blue')
plt.xlabel("Time, $t (s)$", fontname="Segoe UI", fontsize=FS, color='blue')
plt.ylabel('Deflection', fontname="Segoe UI", fontsize=FS, color='green')

# plt.xlim(0, 10)
# plt.ylim(0, 5)
# plt.axis([0, 10, 0, 5])

# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['0', '', '2', '', '4', '', '6', '', '8', '', r'$+\pi$'])
# plt.yticks([0, 1, 2, 3, 4, 5], ['0', '1', '2', '3', '4', '5'], fontsize=FS, color='k')

plt.xticks(fontname="Segoe UI", fontsize=FS, color='k')
plt.yticks(fontname="Segoe UI", fontsize=FS, color='blue')

fig.savefig('Transient3D_Timo', dpi=1000)

plt.show()

# Plot the applied loading versus the time t
fig = plt.figure(999)
FS = 12
print(Time.shape)
print(Fs[dof_output, :].shape)
plt.plot(Time, Fs[dof_output, :], 'b--', linewidth=1.0)
plt.title('Applied loading versus time', fontname="Segoe UI", fontsize=FS + 2, color='red')
plt.xlabel("Time, $t (s)$", fontname="Segoe UI", fontsize=FS, color='blue')
plt.ylabel("Applied Loading, $F (s)$", fontname="Segoe UI", fontsize=FS, color='green')

# plt.xlim(0, 10)
# plt.ylim(0, 5)
# plt.axis([0, 10, 0, 5])

# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['0', '', '2', '', '4', '', '6', '', '8', '', r'$+\pi$'])
# plt.yticks([0, 1, 2, 3, 4, 5], ['0', '1', '2', '3', '4', '5'], fontsize=FS, color='k')

plt.xticks(fontname="Segoe UI", fontsize=FS, color='k')
plt.yticks(fontname="Segoe UI", fontsize=FS, color='blue')

fig.savefig('Load versus time - 3D', dpi=1000)

plt.show()