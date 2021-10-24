import math
import numpy as np
import matplotlib.pyplot as plt
import sys

"""
#===============================================================#
# Purpose:                                                      #
# Draw the section force diagrams of a two dimentional beam     #
# element.                                                      #
#                                                               #
# Syntax:                                                       #
#      eldia2(ex, ey, es, eci, magnfac, magnitude)              #
#                                                               #
# Input:                                                        #
#      ex, ey: ......... nen: number of element nodes           #
#                        nel: number of elements                #
#      es, eci, magnfac, magnitude                              #     
#                                                               #
#===============================================================# 
"""

def eldia22(ex, ey, es, eci, magnfac, magnitude):
    b = np.array(([ex[1] - ex[0], ey[1] - ey[0]]), dtype=float)
    L = np.sqrt(np.dot(b.T, b))
    n = b/L
    es = np.array(es/magnfac)
    """
    N = magnitude[0]
    LL = N/magnfac
    x = magnitude[1]
    y = magnitude[2]
    plt.plot([x, x+LL], [y, y])
    plt.plot([x, x], [y - LL/20, y + LL/20])
    plt.plot([x + LL/2, x + LL/2], [y - LL/20, y + LL/20])
    plt.plot([x + LL, x + LL], [y - LL/20, y + LL/20])
    """
    Nbr = eci.size
    A = np.zeros((Nbr, 2))
    A[0, 0] = ex[0]
    A[0, 1] = ey[0]
    for i in range(1, Nbr):
        A[i,0] = A[0,0] + np.dot(eci[i], n[0])
        A[i,1] = A[0,1] + np.dot(eci[i], n[1])

    B = A
    for i in range(Nbr):
        A[i,0] += np.dot(es[i], n[1])
        A[i,1] -= np.dot(es[i], n[0])

    plt.axis('equal')
    plt.axis('scaled')
    for i in range(Nbr):
        plt.plot([B[i,0], A[i,0]], [B[i,1], A[i,1]])
    plt.plot(A[:,0], A[:,1], color = 'b', linewidth = 1)
    plt.plot([ex[0], A[0,0]], [ey[0], A[0,1]], color = 'b', linewidth = 1)
    plt.plot([ex[1], A[Nbr-1, 0]], [ey[1], A[Nbr-1,1]], color = 'b', linewidth = 1)

    plt.plot(ex.T, ey.T, color = 'r', linewidth = 3)
    return






