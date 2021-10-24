"""
#=========================================================================#
# This code was programmed by:                                            #
# 1. Nguyen Anh Khoa - 1810240                                            #
# 2. Trang Si Tan Khang - 1810215                                         #
# 3. Phan Vuong Phu - 1710235                                             #
# Transient analysis for plane frame structure                            #
# Cross section area = 400x400mm = 160000mm2                              #
# Length of each beam = 6000mm = 6m                                       #
# Length of each column = 3500mm = 3.5m                                   #
# Young's Elastic Modulus = E = 32.5x10^3 MPa = 32500 N/mm2               #
#=========================================================================#
#                                                                         #
#                  __     |------------6m---------|                       #
#                  I    3 ========================= 4                     #
#                  I      H         (6)           H                       #
#                 3.5m    H                       H                       #
#                  I      H (2)                   H  (4)                  #
#                  I      H                       H                       #
#                  __   2 ========================= 5                     #
#                  I      H         (5)           H                       #
#                  I      H                       H                       #
#                 3.5m    H (1)                   H  (3)                  #
#                  I      H                       H                       #
#                  I    ,_H_,                   ,_H_,                     #
#                  __   1                           6                     #
#                                                                         #
#=========================================================================#
"""

import math
import numpy as np
from scipy.linalg import eigh

from mydef.feframe3 import *
from mydef.feasmbl1 import *
from mydef.eldia22 import *
from mydef.Newmark import *


nel = 6                        # number of elements
nnel = 2                       # number of nodes each element
ndof = 3                       # number of degrees of freedom per node
nnode = (nnel - 1)*nel         # total number of nodes
sdof = nnode*ndof              # total number of degrees of freedom

# Young's elastic modulus (N/m^2)
el = 3.25e10
# Cross-sectional area (m^2)
area = 0.16
#shear modulus
Gl=1.354e10
#shear correction factor
ks=1
# Moment of inertia (m^4)
xi = 0.0256/12
# Material density (N/m^3)
rho = 25000

# Nodal coordinate, in [x1,  x2,  x3,  x4,  x5,  x6]  [ y1,  y2,  y3,  y4,  y5,  y6]
x = np.array([0,   0,   0,   6,   6,   6], dtype=float)
y = np.array([0, 3.5,   7,   7, 3.5,   0], dtype=float)
# Element connection, 1:1-2; 2:2-3; 3:6-5; 4:5-4; 5:2-5; 6:3-4
element = np.array(([[1, 2, 6, 5, 2, 3],[2, 3, 5, 4, 5, 4]]), dtype = int) -1

# Apply boundary condition
bcdof = np.zeros((nel*3), dtype=int)
bcval = np.zeros((nel*3), dtype=int)
# Node 1:
bcdof[0] = 0
bcval[0] = 0
bcdof[1] = 1
bcval[1] = 0
bcdof[2] = 2
bcval[2] = 0
# Node 6:
bcdof[15] = 15
bcval[15] = 0
bcdof[16] = 16
bcval[16] = 0
bcdof[17] = 17
bcval[17] = 0

# Force matrix
ff = np.zeros((sdof, 1), dtype=float)
# Stiffness matrix
kk = np.zeros((sdof, sdof), dtype=float)
# Mass matrix
mm = np.zeros((sdof, sdof), dtype=float)
# Horizontal force at node 3 and 4 with value of 10kN
#ff[6] = 10000
#ff[9] = 10000

for iel in range(nel):
    # Dof of each element
    index = np.array(([[(element[0,iel]+1)*3-2, (element[0,iel]+1)*3-1, (element[0,iel]+1)*3],
                       [(element[1,iel]+1)*3-2, (element[1,iel]+1)*3-1, (element[1,iel]+1)*3]]), dtype = int)-1
    # Coordinate of node 1
    x1 = x[element[0,iel]]
    y1 = y[element[0,iel]]
    # Coordinate of node 2
    x2 = x[element[1,iel]]
    y2 = y[element[1,iel]]
    # Length of element 'iel'
    leng = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if (x2 - x1) == 0:
        if  y2 > y1:
            beta = math.pi/2
        else:
            beta = -math.pi/2
    else:
        beta = math.atan((y2-y1)/(x2-x1))
    # Calculate local stiffness matrix
    (k, m) = feframe3 (el,Gl,ks, xi, leng, area, rho, beta, 1)
    # Merge into global stiffness matrix
    kk = feasmbl1(kk, k, index)
    mm = feasmbl1(mm, m, index)

# apply boundary
bcd = np.array(([0, 1, 2, 15, 16, 17]), dtype = int)

# remove the fixed degrees of freedom
free_dof = np.setdiff1d(np.arange(0, sdof), bcd)
# Solve eigenvalue problem
(eigen_evalues, eigen_evectors) = eigh(kk[np.ix_(free_dof, free_dof)], mm[np.ix_(free_dof, free_dof)])
frequencies = np.sqrt(eigen_evalues)/2/math.pi
print(frequencies)

Omega = math.sqrt(eigen_evalues[0])

full_eigen_evectors = np.zeros((sdof, eigen_evectors.shape[1]), dtype = float)
full_eigen_evectors[free_dof, :] = eigen_evectors

"""
#=========================================================================#
# TRANSIENT ANALYSIS USING THE NEWMARK METHOD                             #
#=========================================================================#
"""

# Define the dof of a node to display result
dof_output = 3
measuring_dofs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=int)
# Define applied loads at dofs
ff[3] = 1e4 # N
ff[4] = 1e4 # N
ff[6] = 1e4 # N
ff[9] = 1e4 # N
ff[12] = 1e4 # N

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

fig.savefig('filename_Figure', dpi=1000)

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

plt.xticks(fontname="Times New Roman", fontsize=FS, color='k')
plt.yticks(fontname="Times New Roman", fontsize=FS, color='blue')

fig.savefig('filename_Figure1', dpi=1000)

plt.show()