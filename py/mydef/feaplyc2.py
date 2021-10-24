import math
import numpy as np

"""
#===============================================================#
# Purpose:                                                      #
# Apply constraints to matrix equation [kk]{x}={ff}             #
#                                                               #
# Syntax:                                                       #
# [kk, ff]=feaplybc(kk, ff, bcdof, bcval)                       #
#                                                               #
# Description:                                                  #
#      kk - system matrix before applying constraints           #
#      ff - system vector before applying constraints           #
#      bcdof - a vector containing constrained dof              #
#      bcval - a vector containing contained value              #
#===============================================================# 
"""

def feaplyc2(kk, ff, bcdof, bcval):
    n = len(bcdof)
    sdof = len(kk[0])
    for i in range(n):
        c = bcdof[i]

        for j in range(sdof):
            kk[c, j] = 0

        kk[c, c] = 1
        ff[c] = bcval[i]

    return (kk, ff)