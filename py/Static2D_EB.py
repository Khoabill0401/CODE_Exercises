"""
#=========================================================================#
# This code was programmed by:                                            #
# 1. Nguyen Anh Khoa - 1810240                                            #
# 2. Trang Si Tan Khang - 1810215                                         #
# 3. Phan Vuong Phu - 1710235                                             #
# Static analysis for plane frame structure                               #
# Cross section area = 400x400mm = 160000mm2                              #
# Length of each beam = 6000mm = 6m                                       #
# Length of each column = 3500mm = 3.5m                                   #
# Young's Elastic Modulus = E = 32.5x10^3 MPa = 32500 N/mm2               #
#=========================================================================#
#                                                                         #
#        10kN ---> __     |------------6m---------| ----> 10kN            #
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
import matplotlib as mpl
import matplotlib.pyplot as plt

from mydef.feframe2 import *
from mydef.feasmbl1 import *
from mydef.feaplyc2 import *
from mydef.extract import *
from mydef.beam2s import *
from mydef.eldraw2 import *
from mydef.eldisp2 import *
from mydef.eldia2 import *
from mydef.eldia22 import *


nel = 6                        # number of elements
nnel = 2                       # number of nodes each element
ndof = 3                       # number of degrees of freedom per node
nnode = (nnel - 1)*nel         # total number of nodes
sdof = nnode*ndof              # total number of degrees of freedom

# Young's elastic modulus (N/m^2)
el = 3.25e10

# Cross-sectional area (m^2)
area = 0.16

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
bcdof = np.zeros((nnode*3), dtype=int)
bcval = np.zeros((nnode*3), dtype=int)
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
# Horizontal force at node 3 and 4 with value of 10kN
ff[6] = 10000
ff[9] = 10000

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
    (k, m) = feframe2(el, xi, leng, area, rho, beta, 1)
    # Merge into global stiffness matrix
    kk = feasmbl1(kk, k, index)
# Apply boundary condition
(kk, ff) = feaplyc2(kk, ff, bcdof, bcval)
xxx = np.linalg.inv(kk)
# Calculate element node displacement matrix
fsol = np.dot(np.linalg.inv(kk), ff)
# Print displacements results
for num in range(nnode):
    print("Chuyen vi cua nut thu ", num + 1, " la: [",fsol[[(num+1)*3-2-1]]," ",fsol[[(num+1)*3-1-1]]," ",fsol[[(num+1)*3-1]],"].")

"""
#=========================================================================#
#                      Plot displacements, BMD and SFD                    # 
#=========================================================================#
"""
# Element connection, 1:1-2; 2:2-3; 3:6-5; 4:5-4; 5:2-5; 6:3-4
Edof = np.array(([[0,  0,  1,  2,  3,  4,  5],
                  [1,  3,  4,  5,  6,  7,  8],
                  [2, 15, 16, 17, 12, 13, 14],
                  [3, 12, 13, 14,  9, 10, 11],
                  [4,  3,  4,  5, 12, 13, 14],
                  [5,  6,  7,  8,  9, 10, 11]]), dtype = int)

ed = extract(Edof, fsol)

eq = np.array(([0, 0]), dtype = float)

#for i in range(nel):
#    (es[i], edi[i], eci[i]) = beam2s(i, el, area, xi, x, y, element, ed, eq, 2)
(es0, edi0, eci0) = beam2s(0, el, area, xi, x, y, element, ed[0,:], eq)
(es1, edi1, eci1) = beam2s(1, el, area, xi, x, y, element, ed[1,:], eq)
(es2, edi2, eci2) = beam2s(2, el, area, xi, x, y, element, ed[2,:], eq)
(es3, edi3, eci3) = beam2s(3, el, area, xi, x, y, element, ed[3,:], eq)
(es4, edi4, eci4) = beam2s(4, el, area, xi, x, y, element, ed[4,:], eq)
(es5, edi5, eci5) = beam2s(5, el, area, xi, x, y, element, ed[5,:], eq)

#PlotModel()
"""
# Example for ploting 
t = np.arange(0.0, 2.0, 0.01)
s1 = np.sin(2*np.pi*t)
s2 = np.sin(4*np.pi*t)
plt.figure(1)
plt.subplot(211)
plt.plot(t, s1)
plt.subplot(212)
plt.plot(t, 2*s1)
plt.figure(2)
plt.plot(t, s2)
plt.figure(1)
plt.subplot(211)
plt.plot(t, s2, 's')
ax = plt.gca()
ax.set_xticklabels([])
plt.show()
"""

# Draw displacement graph
#plt.figure(1)
fig = plt.figure('Displacement')
plotpar = ['k', 'b', '*']
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

for i in range(nel):
    eldraw2(x[element[:,i]], y[element[:,i]], plotpar)
plt.title('2D-Frame Model', fontname = 'Segoe UI', fontsize = 15, color = 'b')
plt.xlabel(r"$x$ (m)", fontname="Segoe UI", fontsize = 12, color='k')
plt.ylabel(r"$y$ (m)", fontname="Segoe UI", fontsize = 12, color='k')

plotpar = ['--', 'b', '*']
for i in range(nel):
    eldisp2(x[element[:,i]], y[element[:,i]], ed[i,:], plotpar, 100)

# Draw shear force diagram

fig = plt.figure('Shear force diagram')
plotpar = ['k', 'b', '*']
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

magnitude = np.array([1, 0, 0])
magnfac = eldia2(x[element[:, 2]], y[element[:, 2]], es2[1, :], eci2, 1, magnitude)
eldia22(x[element[:, 0]], y[element[:, 0]], es0[1, :], eci0, magnfac, magnitude)
eldia22(x[element[:, 1]], y[element[:, 1]], es1[1, :], eci1, magnfac, magnitude)
eldia22(x[element[:, 2]], y[element[:, 2]], es2[1, :], eci2, magnfac, magnitude)
eldia22(x[element[:, 3]], y[element[:, 3]], es3[1, :], eci3, magnfac, magnitude)
eldia22(x[element[:, 4]], y[element[:, 4]], es4[1, :], eci4, magnfac, magnitude)
eldia22(x[element[:, 5]], y[element[:, 5]], es5[1, :], eci5, magnfac, magnitude)

plt.title('2D-Frame Model', fontname="Segoe UI", fontsize=15, color='k')
plt.xlabel(r"$x$", fontname="Segoe UI", fontsize=12, color='k')
plt.ylabel(r"$y$", fontname="Segoe UI", fontsize=12, color='k')

plt.xticks(fontname="Segoe UI", fontsize=12, color='k')
plt.yticks(fontname="Segoe UI", fontsize=12, color='k')

# Draw bending moment diagram

fig = plt.figure('Bending moment diagram')
plotpar = ['k', 'b', '*']
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

magnitude = np.array([1, 0, 0])
magnfac = eldia2(x[element[:, 2]], y[element[:, 2]], es2[2, :], eci2, 1, magnitude)
eldia22(x[element[:, 0]], y[element[:, 0]], es0[2, :], eci0, magnfac, magnitude)
eldia22(x[element[:, 1]], y[element[:, 1]], es1[2, :], eci1, magnfac, magnitude)
eldia22(x[element[:, 2]], y[element[:, 2]], es2[2, :], eci2, magnfac, magnitude)
eldia22(x[element[:, 3]], y[element[:, 3]], es3[2, :], eci3, magnfac, magnitude)
eldia22(x[element[:, 4]], y[element[:, 4]], es4[2, :], eci4, magnfac, magnitude)
eldia22(x[element[:, 5]], y[element[:, 5]], es5[2, :], eci5, magnfac, magnitude)

plt.title('2D-Frame Model', fontname="Segoe UI", fontsize=15, color='k')
plt.xlabel(r"$x$", fontname="Segoe UI", fontsize=12, color='k')
plt.ylabel(r"$y$", fontname="Segoe UI", fontsize=12, color='k')

plt.xticks(fontname="Segoe UI", fontsize=12, color='k')
plt.yticks(fontname="Segoe UI", fontsize=12, color='k')

plt.show()





