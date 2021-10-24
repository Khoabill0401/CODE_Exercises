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
# 001. p001_Ackley: Ackley Function                                                                #
# 002. p002_Alpine: Alpine Function                                                                #
# 003. p003_BartelsConn: Bartels Conn Function                                                     #
# 005. p005_Bird: Bird Function                                                                    #
# 006. p006_Bohachevsky: Bohachevsky Function                                                      #
# 007. p007_Booth: Booth Function                                                                  #
# 010. p010_Brent: Brent Function                                                                  #
# 011. p011_Brown: Brown Function                                                                  #
# 012. p012_Bukin: Bukin Function                                                                  #
# 019. p019_CosineMixture: Cosine Mixture Function                                                 #
# 020. p020_Csendes: Csendes Function                                                              #
# 023. p023_Deb: Deb Function                                                                      #
# 030. p030_Exponential: Exponential Function                                                      #
# 032. p032_Griewank: Griewank Function                                                            #
# 037. p037_Hosaki: Hosaki Function                                                                #
# 045. p045_Mishra: Mishra Function                                                                #
# 054. p054_PowellSum: PowellSum Function                                                          #
# 057. p057_Quartic: Quartic Function                                                              #
# 058. p058_Quintic: Quintic Function                                                              #
# 065. p065_Salomon: Salomon Function                                                              #
# 071. p071_SchumerSteiglitz: SchumerSteiglitz Function                                            #
# 075. p075_Sphere: Sphere Function                                                                #
# 076. p076_Step: Step Function                                                                    #
# 077. p077_Stepint: Stepint Function                                                              #
# 079. p079_SumSquares: SumSquares Function                                                        #
# 080. p080_StyblinskiTang: Styblinski - Tang Function                                             #
# 096. p096_XinSheYangSecond: XinSheYangSecond Function                                            #
# 098. p098_XinSheYangFourth: XinSheYangFourth Function                                            #
# 100. p100_Zakharov: Zakharov Function                                                            #
#==================================================================================================#
"""
import matplotlib.pyplot as plt
from py_function.NMRA import *

NMR_pop = 20                     # Number of NMRs
Function_Name = 'p049_Paviani'    # Name of the test function
Max_iteration = 1000           # Maximum number of iterations

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
plt.title('Convergence curve', fontname = "Segoe UI", fontsize=SF, color='k')
plt.xlabel('Iteration', fontname = "Segoe UI", fontsize=SF, color='k')
plt.ylabel('Best score obtained so far', fontname = "Segoe UI", fontsize=SF, color='k')
plt.show()
fig.savefig(filename_Figure, dpi=1000)
# Display the calculation
print('The best solution obtained by NMRA is: ', str(Best_NMRpos))
print('The best optimal value of the objective function found by NMRA is: ', str(Best_NMR))
