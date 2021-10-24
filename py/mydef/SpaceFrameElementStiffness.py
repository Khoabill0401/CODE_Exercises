import math
import numpy as np

"""
#===============================================================#
# Purpose:                                                      #
# This function returns the element stiffness matrix for a      #
# space frame element with modulus of elasticity E, shear       #
# modulus of elasticity G, cross sectional area A, moments of   #
# inertia Iy and Iz, torsional constant J,                      #
# coordinates (x1, y1, z1) for the first node and coordinates   #
# (x2, y2, z2) fpr the second node.                             #
# The size of the element stiffness matrix is 12x12.            #
#                                                               #
#===============================================================# 
"""

def SpaceFrameElementStiffness(el, G, area, yi, zi, J, x_i, y_i, z_i, x_j, y_j, z_j):
    leng = np.sqrt((x_j - x_i) ** 2 + (y_j - y_i) ** 2 + (z_j - z_i) ** 2)
    w1  = el*area/leng
    w2  = 12*el*zi/(leng**3)
    w3  = 6*el*zi/(leng**2)
    w4  = 4*el*zi/leng
    w5  = 2*el*zi/leng
    w6  = 12*el*yi/(leng**3)
    w7  = 6*el*yi/(leng**2)
    w8  = 4*el*yi/leng
    w9  = 2*el*yi/leng
    w10 = G*J/leng
    kprime = np.array(([[ w1,   0,   0,    0,   0,   0, -w1,   0,   0,    0,   0,   0],
                        [  0,  w2,   0,    0,   0,  w3,   0, -w2,   0,    0,   0,  w3],
                        [  0,   0,  w6,    0, -w7,   0,   0,   0, -w6,    0, -w7,   0],
                        [  0,   0,   0,  w10,   0,   0,   0,   0,   0, -w10,   0,   0],
                        [  0,   0, -w7,    0,  w8,   0,   0,   0,  w7,    0,  w9,   0],
                        [  0,  w3,   0,    0,   0,  w4,   0, -w3,   0,    0,   0,  w5],
                        [-w1,   0,   0,    0,   0,   0,  w1,   0,   0,    0,   0,   0],
                        [  0, -w2,   0,    0,   0, -w3,   0,  w2,   0,    0,   0, -w3],
                        [  0,   0, -w6,    0,  w7,   0,   0,   0,  w6,    0,  w7,   0],
                        [  0,   0,   0, -w10,   0,   0,   0,   0,   0,  w10,   0,   0],
                        [  0,   0, -w7,    0,  w9,   0,   0,   0,  w7,    0,  w8,   0],
                        [  0,  w3,   0,    0,   0,  w5,   0, -w3,   0,    0,   0,  w4]]), dtype = float)
    if x_i == x_j and y_i == y_j:
        if z_j > z_i:
            CXx = 0
            CYx = 0
            CZx = 1
            CXy = 0
            CYy = 1
            CZy = 0
            CXz = -1
            CYz = 0
            CZz = 0
        else:
            CXx = 0
            CYx = 0
            CZx = -1
            CXy = 0
            CYy = 1
            CZy = 0
            CXz = 1
            CYz = 0
            CZz = 0
    else:
        CXx = (x_j - x_i)/leng
        CYx = (y_j - y_i)/leng
        CZx = (z_j - z_i)/leng
        D = np.sqrt(CXx**2 + CYx**2)
        CXy = -CYx/D
        CYy = CXx/D
        CZy = 0
        CXz = -CXx*CZx/D
        CYz = -CYx*CZx/D
        CZz = D
    R = np.array(([[CXx, CYx, CZx,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                   [CXy, CYy, CZy,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                   [CXz, CYz, CZz,   0,   0,   0,   0,   0,   0,   0,   0,   0],
                   [  0,   0,   0, CXx, CYx, CZx,   0,   0,   0,   0,   0,   0],
                   [  0,   0,   0, CXy, CYy, CZy,   0,   0,   0,   0,   0,   0],
                   [  0,   0,   0, CXz, CYz, CZz,   0,   0,   0,   0,   0,   0],
                   [  0,   0,   0,   0,   0,   0, CXx, CYx, CZx,   0,   0,   0],
                   [  0,   0,   0,   0,   0,   0, CXy, CYy, CZy,   0,   0,   0],
                   [  0,   0,   0,   0,   0,   0, CXz, CYz, CZz,   0,   0,   0],
                   [  0,   0,   0,   0,   0,   0,   0,   0,   0, CXx, CYx, CZx],
                   [  0,   0,   0,   0,   0,   0,   0,   0,   0, CXy, CYy, CZy],
                   [  0,   0,   0,   0,   0,   0,   0,   0,   0, CXz, CYz, CZz]]), dtype = float)
    k = np.dot(R.T, np.dot(kprime, R))

    return (k)

