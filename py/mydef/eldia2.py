import math
import numpy as np
import matplotlib.pyplot as plt
import sys

"""
#===============================================================#
# Purpose:                                                      #
# Find the magnfac to scale the diagram in right ratio.         #
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

def eldia2(ex, ey, es, eci, magnfac, magnitude):
    b = np.array(([ex[1] - ex[0], ey[1] - ey[0]]), dtype=float)
    L = np.sqrt(np.dot(b.T, b))
    n = b/L
    magnfac = float(abs(max(es))/(0.2*L))

    return (magnfac)






