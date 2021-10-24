import math
import numpy as np

"""
#===============================================================#
# Purpose:                                                      #
# This function returns the element mass matrix for a           #
# space frame element with modulus of elasticity E, shear       #
# modulus of elasticity G, cross sectional area A, moments of   #
# inertia Ix, torsional constant J,                             #
# coordinates (x1, y1, z1) for the first node and coordinates   #
# (x2, y2, z2) fpr the second node.                             #
# The size of the element mass matrix is 12x12.                 #
#                                                               #
#===============================================================# 
"""

def SpaceFrameElementMass(area, rho, xi, x_i, y_i, z_i, x_j, y_j, z_j):
    leng = np.sqrt((x_j - x_i) ** 2 + (y_j - y_i) ** 2 + (z_j - z_i) ** 2)
    a = leng/2
    mass = rho*area*a/105
    r2x = xi/area
    mprime = np.array(([[ 70,     0,     0,       0,       0,       0,  35,     0,     0,       0,       0,       0],
                        [  0,    78,     0,       0,       0,    22*a,   0,    27,     0,       0,       0,   -13*a],
                        [  0,     0,    78,       0,   -22*a,       0,   0,     0,    27,       0,    13*a,       0],
                        [  0,     0,     0,  70*r2x,       0,       0,   0,     0,     0, -35*r2x,       0,       0],
                        [  0,     0, -22*a,       0,  8*a**2,       0,   0,     0, -13*a,       0, -6*a**2,       0],
                        [  0,  22*a,     0,       0,       0,  8*a**2,   0,  13*a,     0,       0,       0, -6*a**2],
                        [ 35,     0,     0,       0,       0,       0,  70,     0,     0,       0,       0,       0],
                        [  0,    27,     0,       0,       0,    13*a,   0,    78,     0,       0,       0,   -22*a],
                        [  0,     0,    27,       0,   -13*a,       0,   0,     0,    78,       0,    22*a,       0],
                        [  0,     0,     0, -35*r2x,       0,       0,   0,     0,     0,  70*r2x,       0,       0],
                        [  0,     0,  13*a,       0, -6*a**2,       0,   0,     0,  22*a,       0,  8*a**2,       0],
                        [  0, -13*a,     0,       0,       0, -6*a**2,   0, -22*a,     0,       0,       0,  8*a**2]]), dtype = float)
    mprime *= mass
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
    mm = np.dot(R.T, np.dot(mprime, R))

    return (mm)

