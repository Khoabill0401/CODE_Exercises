import numpy as np
from py_hybrid_de_function.FEM_10_bar_2D import *

def solve10bar(x):
    NP = 20
    eps1 = 1  # Handle constraints according to dynamic methods
    eps2 = 1.5  # Handle constraints according to dynamic methods
    f = np.zeros((NP, 1), dtype=float)    # objective
    fre = np.zeros((NP, 8), dtype=float)  # frequencies
    c = np.zeros((NP, 10), dtype=float)   # constraints
    Label = '10_bar'

    for i in range(NP):
        #if Label == '10_bar':
        (W, c1, c2, c3, frequencies) = FEM_10_bar_2D(x[:])

        f[i, 0] = W
        c[i, 0:3] = [c1, c2, c3]
        fre[i, :] = frequencies[0:8]

    g = (c >= 0) * c
    # f_penalty = f + 1e5*np.sum(g**2, axis=1).reshape(-1, 1)

    f_penalty = ((1 + eps1 * np.sum(g, axis=1).reshape(-1, 1)) ** eps2) * f

    return (f_penalty)  # , c, fre)

"""
        elif Label == '37_bar':
            (W, c1, c2, c3, frequencies) = FEM_37_bar_2D(x[i, :])

        elif Label == '52_bar':
            (W, c1, c2, c3, frequencies) = FEM_52_bar_3D(x[i, :])

        elif Label == '72_bar':
            (W, c1, c2, c3, frequencies) = FEM_72_bar_3D(x[i, :])

        elif Label == '120_bar':
            (W, c1, c2, c3, frequencies) = FEM_120_bar_3D(x[i, :])

        elif Label == '200_bar':
            (W, c1, c2, c3, frequencies) = FEM_200_bar_2D(x[i, :])
"""
