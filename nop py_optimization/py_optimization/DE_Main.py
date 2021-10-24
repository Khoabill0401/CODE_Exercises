"""
#==================================================================================================#
# Differential Evolution (DE) demo with Appendix A: Test Function Benchmark                        #
# This work has been done by:                                                                      #
# 1. Nguyen Anh Khoa - 1810240                                                                     #
# 2. Phan Vuong Phu - 1710235                                                                      #
# 3. Trang Si Tan Khang - 1810215                                                                  #
#                                                                                                  #
# You can simply define your cost in a separate file and load its handle to fobj                   #
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
from py_function.DE import *

# Name of the test function
Function_name = 'p049_Paviani'
# define population size
pop_size = 10
# define lower and upper bounds for every dimension
# bounds = asarray([(-5.0, 5.0), (-5.0, 5.0)])
# define number of iterations
maxiter = 100
# define scale factor for mutation
F = 0.5
# define crossover rate for recombination
cr = 0.7

# perform differential evolution
solution = DE(Function_name, pop_size, maxiter, F, cr)
#print('\nSolution: f([%s]) = %.5f' % (around(solution[0], decimals=5), solution[1]))

# Draw objective space
Algorithm = 'DE'
Label = Function_name
filename_Figure = Algorithm + '_' + Label + '.pdf'
plt.rc('text', usetex=True)
plt.rc('font', family='Segoe UI')
'----------------------------------------------------------------------------------------------------------------------'
'Plot the convergence history'
fig = plt.figure(filename_Figure)
SF = 14 # Scale factor
# line plot of best objective function values
if solution[3] == 0:
    plt.semilogy(solution[2])
else:
    plt.plot(solution[2])
#plt.plot(solution[2], '.-')
plt.title('Convergence curve', fontname = "Segoe UI", fontsize=SF, color='k')
plt.xlabel('Iteration', fontname = "Segoe UI", fontsize=SF, color='k')
plt.ylabel('Best score obtained so far', fontname = "Segoe UI", fontsize=SF, color='k')
#plt.xlabel('Improvement Number')
#plt.ylabel('Evaluation f(x)')
plt.show()
fig.savefig(filename_Figure, dpi=1000)
print('The best solution obtained by DE is: ', str(solution[1]))
print('The best optimal value of the objective function found by DE is: ', str(solution[0]))
