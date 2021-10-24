import math
import numpy as np

"""
#===============================================================#
# Purpose:                                                      #
# Calculate system matrix in 1D                                 #
#                                                               #
# Syntax:                                                       #
# [index] = feeldof1(iel, nnel, ndof)                           #
#                                                               #
# Description:                                                  #
#      index - vector for dof of each element "iel"             #
#      iel - current element                                    #
#      nnel - element nodes                                     #
#      ndof - element dofs                                      #
#===============================================================# 
"""

def feeldof1(iel, nnel, ndof):
    edof = nnel * ndof
    index = np.zeros((edof), dtype=int)
    start = (iel - 1)*(nnel - 1)*ndof

    for i in range(edof):
        index[i] = start + i

    return (index)