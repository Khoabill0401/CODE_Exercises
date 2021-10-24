import math
import numpy as np

"""
#===============================================================#
# Purpose:                                                      #
# Create Stiffness matrix for Timoshenko beam element           #
#                                                               #
# Syntax:                                                       #
# [k, m] = feframe3 (el,G,ks, xi, leng, area, rho, beta, ipt)   #
#                                                               #
# Description:                                                  #
#      k - element stiffness matrix (size 6x6)                  #
#      m - element mass matrix (size 6x6)                       #
#      el - young's elastic modulus                             #
#      Gl  - shear modulus                                      #
#      ks - shear correction factor                             #
#      xi - moment of inertia                                   #
#      leng - length of element                                 #
#      area - cross sectional area                              #
#      rho - mass density                                       #
#      beta - angular deviation between local and global        #
# coordinate                                                    #
#      ipt = 1 - consistent mass matrix                         #
#          = 2 - lumped mass matrix                             #
#          = 3 - diagonal mass matrix                           #
#===============================================================# 
"""

def feframe3 (el, Gl, ks, xi, leng, area, rho, beta, ipt):
    k = np.zeros((6, 6), dtype = float)
    m = np.zeros((6, 6), dtype = float)
# local stiffness matrix
    y=12*el*xi/(leng**2*Gl*area*ks)
    a=el*area/leng
    b=12*el*xi/(leng**3*(1+y))
    c=6*el*xi/(leng**2*(1+y))
    d=4*el*xi*(1+y/4)/(leng*(1+y))
    e=2*el*xi*(1-y/2)/(leng*(1+y))

    kl = np.array(([[ a,        0,           0, -a,         0,           0],
                    [ 0,        b,           c,  0,        -b,           c],
                    [ 0,        c,           d,  0,        -c,           e],
                    [-a,        0,           0,  a,         0,           0],
                    [ 0,       -b,          -c,  0,         b,          -c],
                    [ 0,        c,           e,  0,        -c,           d]]), dtype = float)
# rotation matrix
    r = np.array(([[ math.cos(beta), math.sin(beta), 0,               0,              0, 0],
                   [-math.sin(beta), math.cos(beta), 0,               0,              0, 0],
                   [              0,              0, 1,               0,              0, 0],
                   [              0,              0, 0,  math.cos(beta), math.sin(beta), 0],
                   [              0,              0, 0, -math.sin(beta), math.cos(beta), 0],
                   [              0,              0, 0,               0,              0, 1]]), dtype = float)
# stiffness matrix at the global axis
    k = np.dot(r.T, np.dot(kl, r))
# consistent mass matrix
    if ipt == 1:
        mm = rho*area*leng/420;
        ma = rho*area*leng/6;
        ml = np.array(([[2*ma,           0,               0,   ma,           0,                0],
                        [   0,      156*mm,      22*leng*mm,    0,       54*mm,      -13*leng*mm],
                        [   0,  22*leng*mm,  4*(leng**2)*mm,    0,  13*leng*mm,  -3*(leng**2)*mm],
                        [  ma,           0,               0, 2*ma,           0,                0],
                        [   0,       54*mm,      13*leng*mm,    0,      156*mm,      -22*leng*mm],
                        [   0, -13*leng*mm, -3*(leng**2)*mm,    0, -22*leng*mm, 4*(leng**2)*mm]]), dtype = float)
# lumped mass matrix
    elif ipt == 2:
        ml = np.zeros((6, 6), dtype = float)
        mass = rho*area*leng
        ml = mass*np.array(([[0.5,   0, 0,   0,   0, 0],
                             [  0, 0.5, 0,   0,   0, 0],
                             [  0,   0, 0,   0,   0, 0],
                             [  0,   0, 0, 0.5,   0, 0],
                             [  0,   0, 0,   0, 0.5, 0],
                             [  0,   0, 0,   0,   0, 0]]), dtype = float)
# diagonal mass matrix
    else:
        ml = np.zeros((6, 6), dtype = float)
        mass = rho*area*leng
        ml = mass*np.array(([[0.5,   0,          0,   0,   0,          0],
                             [  0, 0.5,          0,   0,   0,          0],
                             [  0,   0, leng**2/78,   0,   0,          0],
                             [  0,   0,          0, 0.5,   0,          0],
                             [  0,   0,          0,   0, 0.5,          0],
                             [  0,   0,          0,   0,   0, leng**2/78]]), dtype = float)
# mass in the global system
    m = np.dot(r.T, np.dot(ml, r))

    return (k, m)
