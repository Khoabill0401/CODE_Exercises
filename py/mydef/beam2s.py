import math
import numpy as np

"""
#===============================================================#
# Purpose:                                                      #
# Compute section forces in 2D beam element                     #
#                                                               #
# Input:                                                        #
#      el - Young's elastic modulus                             #
#      area - Cross sectional area                              #
#      xi - Moment of inertia                                   #
#      x, y - Element node coordinates                          #
#      element - Element connections                            #
#      ed = [u1 ... u6] element displacements                   #
#      eq = Distributed loads, local directions                 #
#      num = Number of evaluation points (default = 2)          #
#                                                               #
# Output:                                                       #
#      es = [ N1 V1 M1 ; section forces, local directions, in   #
#             N2 V2 M2 ; num points along the beam, dim(es)=nx3 #
#             .........]                                        #
#      edi = [ u1 v1 ; element displacements, local directions  #
#              u2 v2 ; in num points along the beam, dim(es)=nx2#
#             .......]                                          #
#      eci = [ x1 ; local x - coordinates of the evaluation     #
#              x2 ; points, (x1 = 0 and xn = L)                 #
#             ....]                                             #
#===============================================================# 
"""

def beam2s(no, el, area, xi, x, y, element, ed, eq):
    EA = el*area
    EI = el*xi
    b = np.array(([x[element[1, no]]-x[element[0, no]],
                   y[element[1, no]]-y[element[0, no]]]), dtype=float)
    L = np.sqrt(np.dot(b.T, b))
    qx = eq[0]
    qy = eq[1]
    C = np.array(([[0,      0,    0, 1, 0, 0],
                   [0,      0,    0, 0, 0, 1],
                   [0,      0,    0, 0, 1, 0],
                   [L,      0,    0, 1, 0, 0],
                   [0,   L**3, L**2, 0, L, 1],
                   [0, 3*L**2,  2*L, 0, 1, 0]]), dtype = float)
    n = b/L
    G = np.array(([[ n[0], n[1], 0,     0,    0, 0],
                   [-n[1], n[0], 0,     0,    0, 0],
                   [    0,    0, 1,     0,    0, 0],
                   [    0,    0, 0,  n[0], n[1], 0],
                   [    0,    0, 0, -n[1], n[0], 0],
                   [    0,    0, 0,     0,    0, 1]]), dtype = float)
    #tmp = np.array([0, 0, 0, -qx*(L**2)/(2*EA), qy*(L**4)/(24*EI), qy*(L**3)/(6*EI)])
    M = np.dot(np.linalg.inv(C), (np.dot(G, ed.T))) #- tmp.T))

    A = np.array([M[0], M[3]])
    B = np.array([M[1], M[2], M[4], M[5]])
    #A = np.zeros((2,1))
    #B = np.zeros((4,1))
    #for i in range(2):
     #   A[i,0] = Atmp[i]
    #for i in range(4):
     #   B[i,0] = Btmp[i]

    x = np.array([0, L]).T
    zero = np.zeros((x.size), dtype = float)
    one = np.ones((x.size), dtype = float)

    x2 = np.array([0, L**2])
    x3 = np.array([0, L**3])
    #x4 = np.array([0, L**4])

    u = np.dot(np.array([x, one]), A) #- np.array(x2*qx/(2*EA))

    du = np.dot([one, zero], A) #- np.array(x*qx/EA)

    vtmp = np.array([x3, x2, x, one])
    v = np.dot(vtmp.T, B)  #+ np.array(x4*qy/(24*EI))

    d2vtmp = np.array([6*x, 2*one, zero, zero])
    d2v = np.dot(d2vtmp.T, B) #+ np.array(x2*qy/(2*EI))

    d3vtmp = np.array([6*one, zero, zero, zero])
    d3v = np.dot(d3vtmp.T, B) #+ np.array(x*qy/EI)

    N = EA*du
    M = EI*d2v
    V = -EI*d3v
    es = np.array([N, V, M], dtype = float)
    edi = np.array([u, v], dtype = float)
    eci = x

    return (es, edi, eci)

