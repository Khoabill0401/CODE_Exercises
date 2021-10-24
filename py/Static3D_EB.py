"""
#=========================================================================#
# This code was programmed by:                                            #
# 1. Nguyen Anh Khoa - 1810240                                            #
# 2. Trang Si Tan Khang - 1810215                                         #
# 3. Phan Vuong Phu - 1710235                                             #
# Static analysis for space frame structure                               #
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

from mydef.SpaceFrameElementStiffness import *
from mydef.SpaceFrameAssemble import *
from mydef.feaplyc2 import *
from mydef.SpaceFrameElementForces import *
from mydef.fill_between_3D import *

nel = 11                       # number of elements
nnel = 2                       # number of nodes each element
ndof = 6                       # number of degrees of freedom per node
nnode = 10                     # total number of nodes
sdof = nnode*ndof              # total number of degrees of freedom

# Young's elastic modulus (N/m^2)
el = 3.25e10

# Shear modulus of elasticity (N/m^2)
G = 16250000000 #1.3e10

# Cross-sectional area (m^2)
area = 0.16

# Moment of inertia (m^4)
xi = 0 #0.0256/12
yi = 0.0256/12
zi = 0.0256/12

# Polar moment of inertia (m^4)
J = 0 #3.605e-3

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
        k0 = SpaceFrameElementStiffness(el, G, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k0
    if iel == 1:
        k1 = SpaceFrameElementStiffness(el, G, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k1
    if iel == 2:
        k2 = SpaceFrameElementStiffness(el, G, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k2
    if iel == 3:
        k3 = SpaceFrameElementStiffness(el, G, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k3
    if iel == 4:
        k4 = SpaceFrameElementStiffness(el, G, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k4
    if iel == 5:
        k5 = SpaceFrameElementStiffness(el, G, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k5
    if iel == 6:
        k6 = SpaceFrameElementStiffness(el, G, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k6
    if iel == 7:
        k7 = SpaceFrameElementStiffness(el, G, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k7
    if iel == 8:
        k8 = SpaceFrameElementStiffness(el, G, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k8
    if iel == 9:
        k9 = SpaceFrameElementStiffness(el, G, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k9
    if iel == 10:
        k10 = SpaceFrameElementStiffness(el, G, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j)
        k = k10
    kk = SpaceFrameAssemble(kk, k, element[0, iel], element[1, iel])

ktest = np.zeros((sdof - 24, sdof - 24))
ftest = np.zeros((sdof - 24, 1))
fsol = np.zeros((sdof, 1))
for i in range(sdof - 24):
    ftest[i, 0] = ff[i + 24, 0]
    for j in range(sdof - 24):
        ktest[i, j] = kk[i + 24, j + 24]

# Calculate element node displacement matrix before apply boundary condition
fsoltest = np.dot(np.linalg.inv(ktest), ftest)
# Calculate element node displacement matrix after apply boundary condition
for i in range(sdof - 24):
    fsol[i + 24, 0] = fsoltest[i, 0]

ff = np.dot(kk, fsol)
# Apply boundary condition
(kk, ff) = feaplyc2(kk, ff, bcdof, bcval)

fsol = np.dot(np.linalg.inv(kk), ff)

u = np.zeros((nel, 12), dtype = float)
f = np.zeros((nel, 12), dtype = float)
for iel in range(nel):
    # Coordinate of node i
    x_i = x[element[0, iel]]
    y_i = y[element[0, iel]]
    z_i = z[element[0, iel]]
    # Coordinate of node j
    x_j = x[element[1, iel]]
    y_j = y[element[1, iel]]
    z_j = z[element[1, iel]]
    tmp = np.array(([fsol[element[0, iel]*6], fsol[element[0, iel]*6 + 1], fsol[element[0, iel]*6 + 2], fsol[element[0, iel]*6 + 3], fsol[element[0, iel]*6 + 4], fsol[element[0, iel]*6 + 5],
                     fsol[element[1, iel]*6], fsol[element[1, iel]*6 + 1], fsol[element[1, iel]*6 + 2], fsol[element[1, iel]*6 + 3], fsol[element[1, iel]*6 + 4], fsol[element[1, iel]*6 + 5]]), dtype = float)
    for i in range(12):
        u[iel, i] = tmp[i]
    f[iel] = SpaceFrameElementForces(el, G, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j, u[iel])

"""
#=========================================================================#
#                           Plot Undeformed Shape                         # 
#=========================================================================#
"""

fig = plt.figure('Undeformed Shape', figsize=(5, 5))
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
ax = fig.gca(projection='3d')

#plt.legend()

ax.set_title('3D-Frame model', fontname="Segoe UI", fontsize=12, color='b')
ax.set_xlabel(r"$X$ (m)", fontname="Segoe UI", fontsize=10, color='k')
ax.set_ylabel(r"$Y$ (m)", fontname="Segoe UI", fontsize=10, color='k')
ax.set_zlabel(r"$Z$ (m)", fontname="Segoe UI", fontsize=10, color='k')

for i in range(nel):
    delta = 0.1
    plt.plot(x[element[:, i]], y[element[:, i]], z[element[:, i]], color = 'b', linewidth = 1.2)
    ax.text(x[element[0, i]] + delta, y[element[0, i]] + delta, z[element[0, i]] + delta, element[0, i], size=10, color='m', rotation=-10)
    ax.text(x[element[1, i]] + delta, y[element[1, i]] + delta, z[element[1, i]] + delta, element[1, i], size=10, color='m', rotation=-10)
    ax.scatter(x[element[0, i]], y[element[0, i]], z[element[0, i]], s=15, c='r', marker='o')
    ax.scatter(x[element[1, i]], y[element[1, i]], z[element[1, i]], s=15, c='r', marker='o')
"""
#=========================================================================#
#                            Plot Shear Y Diagram                         # 
#=========================================================================#
"""

fig = plt.figure('Shear Force Diagram in Y Direction', figsize=(5, 5))
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
ax = fig.gca(projection='3d')

#plt.legend()

ax.set_title('3D-Frame model', fontname="Segoe UI", fontsize=12, color='b')
ax.set_xlabel(r"$X$ (m)", fontname="Segoe UI", fontsize=10, color='k')
ax.set_ylabel(r"$Y$ (m)", fontname="Segoe UI", fontsize=10, color='k')
ax.set_zlabel(r"$Z$ (m)", fontname="Segoe UI", fontsize=10, color='k')

for i in range(nel):
    delta = np.array([f[i, 1]/1e4, -f[i, 7]/1e4])
    plt.plot(x[element[:, i]], y[element[:, i]], z[element[:, i]], color = 'b', linewidth = 3)
    xx = [x[element[:, i]], y[element[:, i]], z[element[:, i]]]
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i ==5 or i == 8 or i == 9:
        plt.plot(x[element[:, i]] + delta, y[element[:, i]], z[element[:, i]], color='g', linewidth = 1)
        yy = [x[element[:, i]] + delta, y[element[:, i]], z[element[:, i]]]

    else:
        plt.plot(x[element[:, i]], y[element[:, i]] + delta, z[element[:, i]], color='g', linewidth=1)
        yy = [x[element[:, i]], y[element[:, i]] + delta, z[element[:, i]]]
    fill_between_3D(ax, *xx, *yy, mode = 1)

"""
#=========================================================================#
#                            Plot Shear Z Diagram                         # 
#=========================================================================#
"""

fig = plt.figure('Shear Force Diagram in Z Direction', figsize=(5, 5))
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
ax = fig.gca(projection='3d')

#plt.legend()

ax.set_title('3D-Frame model', fontname="Segoe UI", fontsize=12, color='b')
ax.set_xlabel(r"$X$ (m)", fontname="Segoe UI", fontsize=10, color='k')
ax.set_ylabel(r"$Y$ (m)", fontname="Segoe UI", fontsize=10, color='k')
ax.set_zlabel(r"$Z$ (m)", fontname="Segoe UI", fontsize=10, color='k')

for i in range(nel):
    delta = np.array([f[i, 2]/1e4, -f[i, 8]/1e4])
    plt.plot(x[element[:, i]], y[element[:, i]], z[element[:, i]], color = 'b', linewidth = 3)
    xx = [x[element[:, i]], y[element[:, i]], z[element[:, i]]]
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i ==5 or i == 8 or i == 9:
        plt.plot(x[element[:, i]] + delta, y[element[:, i]], z[element[:, i]], color='g', linewidth = 1)
        yy = [x[element[:, i]] + delta, y[element[:, i]], z[element[:, i]]]

    else:
        plt.plot(x[element[:, i]], y[element[:, i]] + delta, z[element[:, i]], color='g', linewidth=1)
        yy = [x[element[:, i]], y[element[:, i]] + delta, z[element[:, i]]]
    fill_between_3D(ax, *xx, *yy, mode = 1)

"""
#=========================================================================#
#                            Plot Torsion Diagram                         # 
#=========================================================================#
"""

fig = plt.figure('Torsion Diagram', figsize=(5, 5))
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
ax = fig.gca(projection='3d')

#plt.legend()

ax.set_title('3D-Frame model', fontname="Segoe UI", fontsize=12, color='b')
ax.set_xlabel(r"$X$ (m)", fontname="Segoe UI", fontsize=10, color='k')
ax.set_ylabel(r"$Y$ (m)", fontname="Segoe UI", fontsize=10, color='k')
ax.set_zlabel(r"$Z$ (m)", fontname="Segoe UI", fontsize=10, color='k')

for i in range(nel):
    delta = np.array([f[i, 3]/1e4, -f[i, 9]/1e4])
    plt.plot(x[element[:, i]], y[element[:, i]], z[element[:, i]], color = 'b', linewidth = 3)
    xx = [x[element[:, i]], y[element[:, i]], z[element[:, i]]]
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i ==5 or i == 8 or i == 9:
        plt.plot(x[element[:, i]] + delta, y[element[:, i]], z[element[:, i]], color='g', linewidth = 1)
        yy = [x[element[:, i]] + delta, y[element[:, i]], z[element[:, i]]]

    else:
        plt.plot(x[element[:, i]], y[element[:, i]] + delta, z[element[:, i]], color='g', linewidth=1)
        yy = [x[element[:, i]], y[element[:, i]] + delta, z[element[:, i]]]
    fill_between_3D(ax, *xx, *yy, mode = 1)

"""
#=========================================================================#
#                            Plot Moment Y Diagram                        # 
#=========================================================================#
"""

fig = plt.figure('Bending Moment Diagram along Y Axis', figsize=(5, 5))
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
ax = fig.gca(projection='3d')

#plt.legend()

ax.set_title('3D-Frame model', fontname="Segoe UI", fontsize=12, color='b')
ax.set_xlabel(r"$X$ (m)", fontname="Segoe UI", fontsize=10, color='k')
ax.set_ylabel(r"$Y$ (m)", fontname="Segoe UI", fontsize=10, color='k')
ax.set_zlabel(r"$Z$ (m)", fontname="Segoe UI", fontsize=10, color='k')

for i in range(nel):
    delta = np.array([f[i, 4]/1e4, -f[i, 10]/1e4])
    plt.plot(x[element[:, i]], y[element[:, i]], z[element[:, i]], color = 'b', linewidth = 3)
    xx = [x[element[:, i]], y[element[:, i]], z[element[:, i]]]
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i ==5 or i == 8 or i == 9:
        plt.plot(x[element[:, i]] + delta, y[element[:, i]], z[element[:, i]], color='g', linewidth = 1)
        yy = [x[element[:, i]] + delta, y[element[:, i]], z[element[:, i]]]

    else:
        plt.plot(x[element[:, i]], y[element[:, i]] + delta, z[element[:, i]], color='g', linewidth=1)
        yy = [x[element[:, i]], y[element[:, i]] + delta, z[element[:, i]]]
    fill_between_3D(ax, *xx, *yy, mode = 1)

"""
#=========================================================================#
#                            Plot Moment Z Diagram                        # 
#=========================================================================#
"""

fig = plt.figure('Bending Moment Diagram along Z Axis', figsize=(5, 5))
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
ax = fig.gca(projection='3d')

#plt.legend()

ax.set_title('3D-Frame model', fontname="Segoe UI", fontsize=12, color='b')
ax.set_xlabel(r"$X$ (m)", fontname="Segoe UI", fontsize=10, color='k')
ax.set_ylabel(r"$Y$ (m)", fontname="Segoe UI", fontsize=10, color='k')
ax.set_zlabel(r"$Z$ (m)", fontname="Segoe UI", fontsize=10, color='k')

for i in range(nel):
    delta = np.array([f[i, 5]/1e4, -f[i, 11]/1e4])
    plt.plot(x[element[:, i]], y[element[:, i]], z[element[:, i]], color = 'b', linewidth = 3)
    xx = [x[element[:, i]], y[element[:, i]], z[element[:, i]]]
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i ==5 or i == 8 or i == 9:
        plt.plot(x[element[:, i]] + delta, y[element[:, i]], z[element[:, i]], color='g', linewidth = 1)
        yy = [x[element[:, i]] + delta, y[element[:, i]], z[element[:, i]]]

    else:
        plt.plot(x[element[:, i]], y[element[:, i]] + delta, z[element[:, i]], color='g', linewidth=1)
        yy = [x[element[:, i]], y[element[:, i]] + delta, z[element[:, i]]]
    fill_between_3D(ax, *xx, *yy, mode = 1)

plt.show()











