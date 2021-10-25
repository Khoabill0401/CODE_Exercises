import math
import numpy as np
from scipy.linalg import eigh

from py_hybrid_function.Stiff_Mass_Force_Weight import *

def FEM_10_bar_2D(X):
    # Define the problem dimension
    prob_dim = '2D' # 2D or 3D
    # Young's elastic modulus (N/m^2)
    E = 6.98e10
    # Cross-sectional areas (m^2)
    A = np.array((X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8], X[9]), dtype=float)
    # Material density (kg/m^3)
    rho = 2770
    # Nodal coordinate.     [x1,     x2,     x3,   x4,  x5, x6]  [   y1, y2,    y3,   y4,    y5,   y6]
    gcoord = np.array(([[18.288, 18.288, 9.144, 9.144,   0,  0], [9.144,  0, 9.144,    0, 9.144,    0]]), dtype=float)
    # Element connection, 1:3-5; 2:1-3; 3:4-6; 4:...
    element = np.array(([3, 1, 4, 2, 3, 1, 4, 3, 2, 1], [5, 3, 6, 4, 4, 2, 5, 6, 3, 4]), dtype = int)-1
    # total number of elements
    nel = element.shape[1]
    # total number of nodes
    nnode = gcoord.shape[1]
    # number of degree of freedom per node
    ndof = int(2)
    # total degree of freedom of the system
    sdof = nnode*ndof
    # Global stiffness and mass matrixes, force vector
    (K, M, F, W) = Stiff_Mass_Force_Weight(prob_dim, element, gcoord, nel, nnode, sdof, E, A, rho)
    # added non-structural masses
    added_Mass = 454
    for idof in range(0, 8):
        M[idof, idof] += added_Mass
    # apply boundary
    bcdof = np.array(([8, 9, 10, 11]), dtype = int)

    # remove the fixed degrees of freedom
    free_dof = np.setdiff1d(np.arange(0, sdof), bcdof)
    # Solve eigenvalue problem
    (eigen_evalues, eigen_evectors) = eigh(K[np.ix_(free_dof, free_dof)], M[np.ix_(free_dof, free_dof)])
    frequencies = np.sqrt(eigen_evalues)/2/math.pi

    c1 = 7/frequencies[0]-1
    c2 = 15/frequencies[1]-1
    c3 = 20/frequencies[2]-1

    return (W, c1, c2, c3, frequencies)