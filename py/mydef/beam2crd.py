import math
import numpy as np
import matplotlib.pyplot as plt

"""
#===============================================================#
# Purpose:                                                      #
# Calculate the element continuous displacements for a number   #
# of identical 2D Bernoulli beam elements.                      #
#                                                               #
# Syntax:                                                       #
#      beam2crd(ex, ey, ed, mag)                                #                                         #
#                                                               #
# Input:                                                        #
#      ex, ey: ......... nen: number of element nodes           #
#                        nel: number of elements                #
#      ed                                                       #
#      mag                                                      #     
#                                                               #
#===============================================================# 
"""

def beam2crd(ex, ey, ed, mag):
    nie = 1
    ned = len(ed)
    excd = np.zeros((nie, 2))
    eycd = np.zeros((nie, 2))

    for i in range(nie):
        b = np.array([ex[1] - ex[0], ey[1] - ey[0]], dtype=float)
        L = np.sqrt(np.dot(b.T, b))
        n = b/L
        G = np.array(([[n[0], n[1], 0, 0, 0, 0],
                       [-n[1], n[0], 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, n[0], n[1], 0],
                       [0, 0, 0, -n[1], n[0], 0],
                       [0, 0, 0, 0, 0, 1]]), dtype=float)
        d = ed[:].T
        dl = np.dot(G, d)

        xl = np.array([0, L], dtype = float)
        one = np.ones((xl.size), dtype = float)

        cis = np.array([[-1, 1], [L, 0]], dtype = float)/L
        ds = np.array([dl[0], dl[3]])

        ul = np.dot(np.dot(np.array(([xl, one]), dtype = float), cis), ds).T

        cib = np.array(([[  12,     6*L, -12,     6*L],
                         [-6*L, -4*L**2, 6*L, -2*L**2],
                         [   0,    L**3,   0,       0],
                         [L**3,       0,   0,       0]]), dtype = float)/L**3

        db = np.array(([dl[1], dl[2], dl[4], dl[5]]), dtype = float)

        xl2 = np.array([0, L**2])
        xl3 = np.array([0, L**3])
        vl1 = np.array([xl3/6, xl2/2, xl, one]).T
        vl = np.dot(np.dot(vl1, cib),db).T

        cld = np.array([ul, vl], dtype= float)
        A = np.array(([[n[0], -n[1]],
                       [n[1], n[0]]]), dtype = float)
        cd = np.dot(A, cld)

        tmp = np.array([[ex[0]], [ey[0]]])
        xyc = np.dot(A[:,0], xl.T) + np.dot(tmp, one[:, None].T)

        excd[i, :] = xyc[0, :] + mag*cd[0, :]
        eycd[i, :] = xyc[0, :] + mag*cd[0, :]
    return (excd, eycd)


