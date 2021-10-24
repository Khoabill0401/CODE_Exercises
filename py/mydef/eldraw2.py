import math
import numpy as np
import matplotlib.pyplot as plt

"""
#===============================================================#
# Purpose:                                                      #
# Draw the undeformed 2D mesh for a number of elements of the   #
# beam element.                                                 #
#                                                               #
# Syntax:                                                       #
#      eldraw2(ex, ey, plotpar, elnum)                          #
#      eldraw2(ex, ey, plotpar)                                 #
#      eldraw2(ex, ey)                                          #
#                                                               #
# Input:                                                        #
#      ex, ey: ......... nen: number of element nodes           #
#                        nel: number of elements                #
#      plotpar = [linetype, linecolor, modemark]                #
#      elnum = edof[:,0] ; i.e the first column in the topology #
# matrix                                                        #     
#                                                               #
#===============================================================# 
"""

def eldraw2(ex, ey, plotpar):
    x = ex.T
    y = ey.T
    xc = x
    yc = y

    # plot commands
    plt.axis('equal')
    plt.axis('scaled')
    plt.plot(xc, yc, plotpar[0], linewidth = 1.5 )
