import math
import numpy as np

"""
#===============================================================#
# Purpose:                                                      #
# This function assembles the element stiffness matrix k of     #
# the space frame element with nodes i and j into the global    #
# stiffness matrix K. This function returns the global         #
# stiffness matrix K after the element stiffness matrix k      #
# is assembled.                                                 #
#                                                               #
#===============================================================# 
"""

def SpaceFrameAssemble(K, k, i, j):
    K[6 * (i + 1) - 6, 6 * (i + 1) - 6] += k[0, 0]
    K[6 * (i + 1) - 6, 6 * (i + 1) - 5] += k[0, 1]
    K[6 * (i + 1) - 6, 6 * (i + 1) - 4] += k[0, 2]
    K[6 * (i + 1) - 6, 6 * (i + 1) - 3] += k[0, 3]
    K[6 * (i + 1) - 6, 6 * (i + 1) - 2] += k[0, 4]
    K[6 * (i + 1) - 6, 6 * (i + 1) - 1] += k[0, 5]

    K[6 * (i + 1) - 6, 6 * (j + 1) - 6] += k[0, 6]
    K[6 * (i + 1) - 6, 6 * (j + 1) - 5] += k[0, 7]
    K[6 * (i + 1) - 6, 6 * (j + 1) - 4] += k[0, 8]
    K[6 * (i + 1) - 6, 6 * (j + 1) - 3] += k[0, 9]
    K[6 * (i + 1) - 6, 6 * (j + 1) - 2] += k[0, 10]
    K[6 * (i + 1) - 6, 6 * (j + 1) - 1] += k[0, 11]

    K[6 * (i + 1) - 5, 6 * (i + 1) - 6] += k[1, 0]
    K[6 * (i + 1) - 5, 6 * (i + 1) - 5] += k[1, 1]
    K[6 * (i + 1) - 5, 6 * (i + 1) - 4] += k[1, 2]
    K[6 * (i + 1) - 5, 6 * (i + 1) - 3] += k[1, 3]
    K[6 * (i + 1) - 5, 6 * (i + 1) - 2] += k[1, 4]
    K[6 * (i + 1) - 5, 6 * (i + 1) - 1] += k[1, 5]

    K[6 * (i + 1) - 5, 6 * (j + 1) - 6] += k[1, 6]
    K[6 * (i + 1) - 5, 6 * (j + 1) - 5] += k[1, 7]
    K[6 * (i + 1) - 5, 6 * (j + 1) - 4] += k[1, 8]
    K[6 * (i + 1) - 5, 6 * (j + 1) - 3] += k[1, 9]
    K[6 * (i + 1) - 5, 6 * (j + 1) - 2] += k[1, 10]
    K[6 * (i + 1) - 5, 6 * (j + 1) - 1] += k[1, 11]

    K[6 * (i + 1) - 4, 6 * (i + 1) - 6] += k[2, 0]
    K[6 * (i + 1) - 4, 6 * (i + 1) - 5] += k[2, 1]
    K[6 * (i + 1) - 4, 6 * (i + 1) - 4] += k[2, 2]
    K[6 * (i + 1) - 4, 6 * (i + 1) - 3] += k[2, 3]
    K[6 * (i + 1) - 4, 6 * (i + 1) - 2] += k[2, 4]
    K[6 * (i + 1) - 4, 6 * (i + 1) - 1] += k[2, 5]

    K[6 * (i + 1) - 4, 6 * (j + 1) - 6] += k[2, 6]
    K[6 * (i + 1) - 4, 6 * (j + 1) - 5] += k[2, 7]
    K[6 * (i + 1) - 4, 6 * (j + 1) - 4] += k[2, 8]
    K[6 * (i + 1) - 4, 6 * (j + 1) - 3] += k[2, 9]
    K[6 * (i + 1) - 4, 6 * (j + 1) - 2] += k[2, 10]
    K[6 * (i + 1) - 4, 6 * (j + 1) - 1] += k[2, 11]

    K[6 * (i + 1) - 3, 6 * (i + 1) - 6] += k[3, 0]
    K[6 * (i + 1) - 3, 6 * (i + 1) - 5] += k[3, 1]
    K[6 * (i + 1) - 3, 6 * (i + 1) - 4] += k[3, 2]
    K[6 * (i + 1) - 3, 6 * (i + 1) - 3] += k[3, 3]
    K[6 * (i + 1) - 3, 6 * (i + 1) - 2] += k[3, 4]
    K[6 * (i + 1) - 3, 6 * (i + 1) - 1] += k[3, 5]

    K[6 * (i + 1) - 3, 6 * (j + 1) - 6] += k[3, 6]
    K[6 * (i + 1) - 3, 6 * (j + 1) - 5] += k[3, 7]
    K[6 * (i + 1) - 3, 6 * (j + 1) - 4] += k[3, 8]
    K[6 * (i + 1) - 3, 6 * (j + 1) - 3] += k[3, 9]
    K[6 * (i + 1) - 3, 6 * (j + 1) - 2] += k[3, 10]
    K[6 * (i + 1) - 3, 6 * (j + 1) - 1] += k[3, 11]

    K[6 * (i + 1) - 2, 6 * (i + 1) - 6] += k[4, 0]
    K[6 * (i + 1) - 2, 6 * (i + 1) - 5] += k[4, 1]
    K[6 * (i + 1) - 2, 6 * (i + 1) - 4] += k[4, 2]
    K[6 * (i + 1) - 2, 6 * (i + 1) - 3] += k[4, 3]
    K[6 * (i + 1) - 2, 6 * (i + 1) - 2] += k[4, 4]
    K[6 * (i + 1) - 2, 6 * (i + 1) - 1] += k[4, 5]

    K[6 * (i + 1) - 2, 6 * (j + 1) - 6] += k[4, 6]
    K[6 * (i + 1) - 2, 6 * (j + 1) - 5] += k[4, 7]
    K[6 * (i + 1) - 2, 6 * (j + 1) - 4] += k[4, 8]
    K[6 * (i + 1) - 2, 6 * (j + 1) - 3] += k[4, 9]
    K[6 * (i + 1) - 2, 6 * (j + 1) - 2] += k[4, 10]
    K[6 * (i + 1) - 2, 6 * (j + 1) - 1] += k[4, 11]

    K[6 * (i + 1) - 1, 6 * (i + 1) - 6] += k[5, 0]
    K[6 * (i + 1) - 1, 6 * (i + 1) - 5] += k[5, 1]
    K[6 * (i + 1) - 1, 6 * (i + 1) - 4] += k[5, 2]
    K[6 * (i + 1) - 1, 6 * (i + 1) - 3] += k[5, 3]
    K[6 * (i + 1) - 1, 6 * (i + 1) - 2] += k[5, 4]
    K[6 * (i + 1) - 1, 6 * (i + 1) - 1] += k[5, 5]

    K[6 * (i + 1) - 1, 6 * (j + 1) - 6] += k[5, 6]
    K[6 * (i + 1) - 1, 6 * (j + 1) - 5] += k[5, 7]
    K[6 * (i + 1) - 1, 6 * (j + 1) - 4] += k[5, 8]
    K[6 * (i + 1) - 1, 6 * (j + 1) - 3] += k[5, 9]
    K[6 * (i + 1) - 1, 6 * (j + 1) - 2] += k[5, 10]
    K[6 * (i + 1) - 1, 6 * (j + 1) - 1] += k[5, 11]

    K[6 * (j + 1) - 6, 6 * (i + 1) - 6] += k[6, 0]
    K[6 * (j + 1) - 6, 6 * (i + 1) - 5] += k[6, 1]
    K[6 * (j + 1) - 6, 6 * (i + 1) - 4] += k[6, 2]
    K[6 * (j + 1) - 6, 6 * (i + 1) - 3] += k[6, 3]
    K[6 * (j + 1) - 6, 6 * (i + 1) - 2] += k[6, 4]
    K[6 * (j + 1) - 6, 6 * (i + 1) - 1] += k[6, 5]

    K[6 * (j + 1) - 6, 6 * (j + 1) - 6] += k[6, 6]
    K[6 * (j + 1) - 6, 6 * (j + 1) - 5] += k[6, 7]
    K[6 * (j + 1) - 6, 6 * (j + 1) - 4] += k[6, 8]
    K[6 * (j + 1) - 6, 6 * (j + 1) - 3] += k[6, 9]
    K[6 * (j + 1) - 6, 6 * (j + 1) - 2] += k[6, 10]
    K[6 * (j + 1) - 6, 6 * (j + 1) - 1] += k[6, 11]

    K[6 * (j + 1) - 5, 6 * (i + 1) - 6] += k[7, 0]
    K[6 * (j + 1) - 5, 6 * (i + 1) - 5] += k[7, 1]
    K[6 * (j + 1) - 5, 6 * (i + 1) - 4] += k[7, 2]
    K[6 * (j + 1) - 5, 6 * (i + 1) - 3] += k[7, 3]
    K[6 * (j + 1) - 5, 6 * (i + 1) - 2] += k[7, 4]
    K[6 * (j + 1) - 5, 6 * (i + 1) - 1] += k[7, 5]

    K[6 * (j + 1) - 5, 6 * (j + 1) - 6] += k[7, 6]
    K[6 * (j + 1) - 5, 6 * (j + 1) - 5] += k[7, 7]
    K[6 * (j + 1) - 5, 6 * (j + 1) - 4] += k[7, 8]
    K[6 * (j + 1) - 5, 6 * (j + 1) - 3] += k[7, 9]
    K[6 * (j + 1) - 5, 6 * (j + 1) - 2] += k[7, 10]
    K[6 * (j + 1) - 5, 6 * (j + 1) - 1] += k[7, 11]

    K[6 * (j + 1) - 4, 6 * (i + 1) - 6] += k[8, 0]
    K[6 * (j + 1) - 4, 6 * (i + 1) - 5] += k[8, 1]
    K[6 * (j + 1) - 4, 6 * (i + 1) - 4] += k[8, 2]
    K[6 * (j + 1) - 4, 6 * (i + 1) - 3] += k[8, 3]
    K[6 * (j + 1) - 4, 6 * (i + 1) - 2] += k[8, 4]
    K[6 * (j + 1) - 4, 6 * (i + 1) - 1] += k[8, 5]

    K[6 * (j + 1) - 4, 6 * (j + 1) - 6] += k[8, 6]
    K[6 * (j + 1) - 4, 6 * (j + 1) - 5] += k[8, 7]
    K[6 * (j + 1) - 4, 6 * (j + 1) - 4] += k[8, 8]
    K[6 * (j + 1) - 4, 6 * (j + 1) - 3] += k[8, 9]
    K[6 * (j + 1) - 4, 6 * (j + 1) - 2] += k[8, 10]
    K[6 * (j + 1) - 4, 6 * (j + 1) - 1] += k[8, 11]

    K[6 * (j + 1) - 3, 6 * (i + 1) - 6] += k[9, 0]
    K[6 * (j + 1) - 3, 6 * (i + 1) - 5] += k[9, 1]
    K[6 * (j + 1) - 3, 6 * (i + 1) - 4] += k[9, 2]
    K[6 * (j + 1) - 3, 6 * (i + 1) - 3] += k[9, 3]
    K[6 * (j + 1) - 3, 6 * (i + 1) - 2] += k[9, 4]
    K[6 * (j + 1) - 3, 6 * (i + 1) - 1] += k[9, 5]

    K[6 * (j + 1) - 3, 6 * (j + 1) - 6] += k[9, 6]
    K[6 * (j + 1) - 3, 6 * (j + 1) - 5] += k[9, 7]
    K[6 * (j + 1) - 3, 6 * (j + 1) - 4] += k[9, 8]
    K[6 * (j + 1) - 3, 6 * (j + 1) - 3] += k[9, 9]
    K[6 * (j + 1) - 3, 6 * (j + 1) - 2] += k[9, 10]
    K[6 * (j + 1) - 3, 6 * (j + 1) - 1] += k[9, 11]

    K[6 * (j + 1) - 2, 6 * (i + 1) - 6] += k[10, 0]
    K[6 * (j + 1) - 2, 6 * (i + 1) - 5] += k[10, 1]
    K[6 * (j + 1) - 2, 6 * (i + 1) - 4] += k[10, 2]
    K[6 * (j + 1) - 2, 6 * (i + 1) - 3] += k[10, 3]
    K[6 * (j + 1) - 2, 6 * (i + 1) - 2] += k[10, 4]
    K[6 * (j + 1) - 2, 6 * (i + 1) - 1] += k[10, 5]

    K[6 * (j + 1) - 2, 6 * (j + 1) - 6] += k[10, 6]
    K[6 * (j + 1) - 2, 6 * (j + 1) - 5] += k[10, 7]
    K[6 * (j + 1) - 2, 6 * (j + 1) - 4] += k[10, 8]
    K[6 * (j + 1) - 2, 6 * (j + 1) - 3] += k[10, 9]
    K[6 * (j + 1) - 2, 6 * (j + 1) - 2] += k[10, 10]
    K[6 * (j + 1) - 2, 6 * (j + 1) - 1] += k[10, 11]

    K[6 * (j + 1) - 1, 6 * (i + 1) - 6] += k[11, 0]
    K[6 * (j + 1) - 1, 6 * (i + 1) - 5] += k[11, 1]
    K[6 * (j + 1) - 1, 6 * (i + 1) - 4] += k[11, 2]
    K[6 * (j + 1) - 1, 6 * (i + 1) - 3] += k[11, 3]
    K[6 * (j + 1) - 1, 6 * (i + 1) - 2] += k[11, 4]
    K[6 * (j + 1) - 1, 6 * (i + 1) - 1] += k[11, 5]

    K[6 * (j + 1) - 1, 6 * (j + 1) - 6] += k[11, 6]
    K[6 * (j + 1) - 1, 6 * (j + 1) - 5] += k[11, 7]
    K[6 * (j + 1) - 1, 6 * (j + 1) - 4] += k[11, 8]
    K[6 * (j + 1) - 1, 6 * (j + 1) - 3] += k[11, 9]
    K[6 * (j + 1) - 1, 6 * (j + 1) - 2] += k[11, 10]
    K[6 * (j + 1) - 1, 6 * (j + 1) - 1] += k[11, 11]
    return (K)