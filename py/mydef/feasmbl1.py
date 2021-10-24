import math
import numpy as np

"""
#===============================================================#
# Purpose:                                                      #
# Assembly of element matrices into the system matrix           #
#                                                               #
# Syntax:                                                       #
# [kk]=feasmbl1(kk, k, index)                                   #
#                                                               #
# Description:                                                  #
#      kk - system matrix                                       #
#      k - element matrix                                       #
#      index - dof vector associated with an element            #
#===============================================================# 
"""

def feasmbl1(kk, k, index):
    n = len(index[:,0])
    m = len(index[0,:])
    inc = np.zeros((n*m), dtype=int)
    start = 0
    for i in range(n):
        for j in range(m):
            inc[start] = index[i, j]
            start += 1
    for i in range(n*m):
        for j in range(n*m):
            kk[inc[i], inc[j]] = kk[inc[i], inc[j]] + k[i, j]

    return kk