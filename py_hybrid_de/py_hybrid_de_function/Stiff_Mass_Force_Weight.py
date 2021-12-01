import math
import numpy as np


def Stiff_Mass_Force_Weight(prob_dim, element, gcoord, nel, nnode, sdof, E, A, rho):
    K = np.zeros((sdof, sdof), dtype = float)
    M = np.zeros((sdof, sdof), dtype=float)
    F = np.zeros((sdof, 1), dtype=float)
    W = 0
    if prob_dim == '2D':
        for e in range(nel):
            nd = element[:, e]
            x = gcoord[0, nd]
            y = gcoord[1, nd]
            le = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
            nx = (x[1] - x[0])/le
            ny = (y[1] - y[0])/le

            T = np.array(([[nx, ny, 0, 0], [0, 0, nx, ny]]), dtype = float)

            Ke_local = (A[e]*E/le)*np.array([[1, -1], [-1, 1]])
            Ke = np.dot(np.dot(T.transpose(), Ke_local), T)

            Me = rho*le*A[e]*np.array([[2, 0, 1, 0], [0, 2, 0, 1], [1, 0, 2, 0], [0, 1, 0, 2]])/6

            elemDof = np.array(([2*nd[0], 2*nd[0]+1,  2*nd[1],  2*nd[1]+1]), dtype = int)

            K[np.ix_(elemDof, elemDof)] += Ke
            M[np.ix_(elemDof, elemDof)] += Me
            W += rho*A[e]*le

    elif prob_dim == '3D':
        for e in range(nel):
            nd = element[:, e]
            x = gcoord[0, nd]
            y = gcoord[1, nd]
            z = gcoord[2, nd]
            le = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2 + (z[1] - z[0])**2)
            nx = (x[1] - x[0])/le
            ny = (y[1] - y[0])/le
            nz = (z[1] - z[0])/le

            T = np.array(([[nx, ny, nz, 0, 0, 0], [0, 0, 0, nx, ny, nz]]), dtype = float)

            Ke_local = (A[e]*E/le)*np.array([[1, -1], [-1, 1]])
            Ke = np.dot(np.dot(T.transpose(), Ke_local), T)

            Me = rho*le*A[e]*np.array([[2, 0, 0, 1, 0, 0], [0, 2, 0, 0, 1, 0], [0, 0, 2, 0, 0, 1], [1, 0, 0, 2, 0, 0], [0, 1, 0, 0, 2, 0], [0, 0, 1, 0, 0, 2]])/6

            elemDof = np.array(([3*nd[0], 3*nd[0]+1,  3*nd[0]+2, 3*nd[1],  3*nd[1]+1, 3*nd[1]+2]), dtype = int)

            K[np.ix_(elemDof, elemDof)] += Ke
            M[np.ix_(elemDof, elemDof)] += Me
            W += rho*A[e]*le

    return (K, M, F, W)