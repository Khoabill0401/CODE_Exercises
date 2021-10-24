"""
#==================================================================================================#
# Naked Mole Rat Algorithm (NMRA) demo with Appendix A: Test Function Benchmark                    #
# This work has been done by:                                                                      #
# 1. Nguyen Anh Khoa - 1810240                                                                     #
# 2. Phan Vuong Phu - 1710235                                                                      #
# 3. Trang Si Tan Khang - 1810215                                                                  #
#                                                                                                  #
# You can simply define your cost in a separate file and load its handle to fobj                   #
# The initial parameters that you need are:                                                        #
# ________________________________________________________________________________________________ #
# fobj = @YourCostFunction                                                                         #
# dim = number of your variables                                                                   #
# Max_iteration = maximum number of generations                                                    #
# NMR_pop = number of search agents                                                                #
# lb = [lb1, lb2, ..., lbn] where lbn is the lower bound of variable n                             #
# ub = [ub1, ub2, ..., ubn] where ubn is the upper bound of variable n                             #
# If all the variables have equal lower bound you can just define lb and ub as two single numbers  #
#                                                                                                  #
# To run NMR_Main:                                                                                 #
# [Best_score, Best_pos, cg_curve] = NMRA(NMR_pop, Max_iteration, lb, ub, dim, fobj)               #
# ________________________________________________________________________________________________ #
# Provided Benchmark functions that we have tested:                                                #
# 1. p001_Ackley: Ackley Function                                                                  #
# 19. p019_CosineMixture: Cosine Mixture Function                                                  #
# 54. p054_PowellSum: Powell Sum Function                                                          #
# 57. p057_Quartic: Quartic Function                                                               #
# 58. p058_Quintic: Quintic Function                                                               #
# 65. p065_Salomon: Salomon Function                                                               #
# 71. p071_SchumerSteiglitz: Schumer Steiglitz Function                                            #
# 75. p075_Sphere: Sphere Function                                                                 #
# 76. p076_Step: Step Function                                                                     #
# 77. p077_Stepint: Stepint Function  Eror                                                         #
# 79. p079_SumSquares: Sum Squares Function                                                        #
# 80. p080_StyblinskiTang: Styblinski - Tang Function                                              #
# 96. p096_XinSheYangSecond:Xin - She Yang Second Function                                         #
# 98. p098_XinSheYangFourth:Xin - She Yang Fourth Function                                         #
# 100. p100_Zakharov: Zakhavor Function                                                            #
#==================================================================================================#
"""
import matplotlib
import matplotlib.pyplot as plt
from py_function.NMRA import *

NMR_pop = 20                     # Number of NMRs
Function_Name = 'p054_PowellSum'    # Name of the test function
Max_iteration = 1000             # Maximum number of iterations

# Run NMR Algorithm
(Best_NMR, Best_NMRpos, cg_curve, ptype) = NMRA(Function_Name, Max_iteration, NMR_pop)
# Draw objective space
Algorithm = 'NMR'
Label = Function_Name
filename_Figure = Algorithm + '_' + Label + '.pdf'
plt.rc('text', usetex=True)
plt.rc('font', family='Segoe UI')
'----------------------------------------------------------------------------------------------------------------------'
'Plot the convergence history'
fig = plt.figure(filename_Figure)
SF = 14 # Scale factor
if ptype == 0:
    plt.semilogy(cg_curve)
else:
    plt.plot(cg_curve)
#plt.xlim(0, Max_iteration)
#plt.ylim(min(cg_curve), max(cg_curve))
#plt.tight_layout()
#plt.autoscale(enable=True, axis='both', tight=True)
#plt.axis('scaled')
#plt.axis('auto')
#plt.axis('tight')
plt.title('Convergence curve', fontname = "Segoe UI", fontsize=SF, color='k')
plt.xlabel('Iteration', fontname = "Segoe UI", fontsize=SF, color='k')
plt.ylabel('Best score obtained so far', fontname = "Segoe UI", fontsize=SF, color='k')
plt.show()
fig.savefig(filename_Figure, dpi=1000)
# Display the calculation
print('The best solution obtained by NMRA is: ', str(Best_NMRpos))
print('The best optimal value of the objective function found by NMRA is: ', str(Best_NMR))
