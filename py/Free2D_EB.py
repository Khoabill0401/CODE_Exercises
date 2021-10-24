"""
#=========================================================================#
# This code was programmed by:                                            #
# 1. Nguyen Anh Khoa - 1810240                                            #
# 2. Trang Si Tan Khang - 1810215                                         #
# 3. Phan Vuong Phu - 1710235                                             #
# Free vibration for plane frame structure                                #
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

from mydef.feframe2 import *
from mydef.feasmbl1 import *
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
    mm = feasmbl1(mm, m, index)

# apply boundary
bcd = np.array(([0, 1, 2, 15, 16, 17]), dtype = int)

# remove the fixed degrees of freedom
free_dof = np.setdiff1d(np.arange(0, sdof), bcd)
# Solve eigenvalue problem
(eigen_evalues, eigen_evectors) = eigh(kk[np.ix_(free_dof, free_dof)], mm[np.ix_(free_dof, free_dof)])
frequencies = np.sqrt(eigen_evalues)/(2*math.pi)
print(frequencies)
#full_eigen_evectors = np.zeros((sdof, eigen_evectors.shape[1]), dtype = float)
#full_eigen_evectors[free_dof, :] = eigen_evectors
#print(full_eigen_evectors)
#print(full_eigen_evectors[:, 0])