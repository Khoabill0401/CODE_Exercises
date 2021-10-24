import math
import numpy as np

# For more information about the algorithm
# Please refer to page 362: Mechanics of laminated composite plates and shells: Theory and analysis, J.N.Reddy
# Please refer to page 324: An Introduction to the Finite Element Method, J.N.Reddy

def Newmark(kk, mm, Fs, sdof, free_dof, dof_output, deltaT, ns, measuring_dofs):
    # Define the coefficents used in the Newmark family
    alpha = 1 / 2
    beta = 1 / 4
    gamma = 2 * beta
    # Check the stability condition for choosing time step deltaT
    #detalCritical = 1/math.sqrt(0.5*Omega*(alpha-gamma))

    # The constant coefficients are used for calculating
    a1 = alpha * deltaT
    a2 = (1 - alpha) * deltaT
    a3 = 1 / (beta * (deltaT ** 2))
    a4 = a3 * deltaT
    a5 = 1 / gamma - 1
    a6 = alpha / (beta * deltaT)
    a7 = alpha / beta - 1
    a8 = (alpha / gamma - 1) * deltaT

    # Define the damping matrix if necessary
    alphaM = 0.05
    alphaK = 0.001
    C = alphaM * mm + alphaK * kk

    # calculate displacement, velocity and acceleration at the time t=0 (s=0)
    s = 0
    Us = np.zeros((sdof, ns), dtype=float)                            # are assumed, please see Eq. (6.2.28b)
    Us1dot = np.zeros((sdof, ns), dtype=float)                        # are assumed, please see Eq. (6.2.28b)
    F0eff = Fs[:, s] - np.dot(mm, Us[:, s]) - np.dot(C, Us1dot[:, s])  # The effective global applied load vector at t = 0
    K0eff = mm                                                         # The effective global mass matrix at t = 0
    Us2dot = np.zeros((sdof, ns), dtype=float)
    Us2dot_free_dof = np.linalg.solve(K0eff[np.ix_(free_dof, free_dof)], F0eff[np.ix_(free_dof)])  # Solve to find the acceleration Us2dot at time t = 0
    Us2dot[free_dof, s] = Us2dot_free_dof

    deflection = np.zeros((ns, 1), dtype=float)
    deflection[s] = Us[dof_output, s]
    # From the second time step to the end one
    for s in range(1, ns):
        As = a3 * Us[:, s - 1] + a4 * Us1dot[:, s - 1] + a5 * Us2dot[:, s - 1]
        Bs = a6 * Us[:, s - 1] + a7 * Us1dot[:, s - 1] + a8 * Us2dot[:, s - 1]

        Feff = Fs[:, s] + np.dot(mm, As) + np.dot(C, Bs)
        Keff = mm + a3 * mm + a6 * C

        Us_free_dof = np.linalg.solve(Keff[np.ix_(free_dof, free_dof)], Feff[np.ix_(free_dof)])
        Us[free_dof, s] = Us_free_dof

        Us2dot[:, s] = a3 * (Us[:, s] - Us[:, s - 1]) - a4 * Us1dot[:, s - 1] - a5 * Us2dot[:, s - 1]
        Us1dot[:, s] = Us1dot[:, s - 1] + a2 * Us2dot[:, s - 1] + a1 * Us2dot[:, s]

        deflection[s] = Us[dof_output, s]

    disp_res_matrix = Us[measuring_dofs, :]     # displacement response matrix w.r.t measuring points
    acce_res_matrix = Us2dot[measuring_dofs, :] # acceleration response matrix w.r.t measuring points

    disp_res_vector = disp_res_matrix.flatten()
    acce_res_vector = acce_res_matrix.flatten()

    print(np.dot(disp_res_vector.transpose(), disp_res_vector))
    print(np.dot(acce_res_vector.transpose(), acce_res_vector))

    return (deflection, disp_res_vector, acce_res_vector)


