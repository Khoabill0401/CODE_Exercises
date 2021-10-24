import math
import numpy as np
import matplotlib.pyplot as plt
from mydef.beam2crd import *

"""
#===============================================================#
# Purpose:                                                      #
# Draw the deformed 2D mesh for a number of elements of the   #
# beam elements                                                 #
#                                                               #
# Syntax:                                                       #
#      eldisp2(ex, ey, ed, plotpar, magnfac)                    #                                    #
#                                                               #
# Input:                                                        #
#      ex, ey: ......... nen: number of element nodes           #
#                        nel: number of elements                #
#      plotpar = [linetype, linecolor, modemark]                #
#      magnfac                                                  #     
#                                                               #
#===============================================================# 
"""

def eldisp2(ex, ey, ed, plotpar, magnfac):
    #dxmax = max(max(ex.T)-min(ex.T))
    #dymax = max(max(ey.T)-min(ey.T))
    #dlmax = max(dxmax, dymax)
    #edmax = max(max(abs(ed)))
    krel = 0.1
    #magnfac = krel*dlmax/edmax

    k = magnfac
    ed1 = np.array([ed[0], ed[3]])
    ed2 = np.array([ed[1], ed[4]])
    x = (ex + k*ed1).T
    y = (ey + k*ed2).T
    (exc, eyc) = beam2crd(ex, ey, ed, k)
    xc = exc
    yc = eyc

    plt.axis('equal')
    plt.axis('scaled')
    plt.plot(xc, yc, plotpar[0], linewidth=1, color='r')
    plt.plot(x, y, plotpar[0], linewidth=1, color='r')