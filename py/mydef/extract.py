import math
import numpy as np

"""
#===============================================================#
# Purpose:                                                      #
# Extract element displacements from the global displacement    #
# vector according to the topology matrix edof.                 #
#                                                               #
# Input:                                                        #
#      fsol - the global displacement vector                    #
#      Edof - topology matrix                                   #         
#                                                               #
# Output:                                                       #
#      ed - element displacement matrix                         #
#===============================================================# 
"""

def extract(Edof, fsol):
    nie = len(Edof)
    n = len(Edof[0])
    ed = np.zeros((nie,n-1), dtype = float)
    t = Edof[:,1:(n-1)]

    for i in range(nie):
        ed[i,0:(n-2)] = fsol[t[i,:]].T

    return ed

