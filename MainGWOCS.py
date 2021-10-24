"""
# ==================================================================================== #
# The purpose of this hybrid GWOCS optimization algorithm is combining the global      #
# converging power of GWO with CS. We tested it on benchmark optimization functions    #
# and found GWOCS performing better than GWO alone.                                    #
# ==================================================================================== #
# This work has been done by:                                                          #
# Nguyen Anh Khoa  - 1810240                                                           #
# Phone: 0868.840.441                                                                  #
# Email: khoa.nguyen41@hcmut.edu.vn                                                    #
# ==================================================================================== #
# Provided Benchmark functions that we have tested:                                    #
# 001. p001_Ackley: Ackley Function                                                    #
# 002. p002_Alpine: Alpine Function                                                    #
# 003. p003_BartelsConn: Bartels Conn Function                                         #
# 005. p005_Bird: Bird Function                                                        #
# 006. p006_Bohachevsky: Bohachevsky Function                                          #
# 007. p007_Booth: Booth Function                                                      #
# 010. p010_Brent: Brent Function                                                      #
# 011. p011_Brown: Brown Function                                                      #
# 012. p012_Bukin: Bukin Function                                                      #
# 019. p019_CosineMixture: Cosine Mixture Function                                     #
# 020. p020_Csendes: Csendes Function                                                  #
# 023. p023_Deb: Deb Function                                                          #
# 030. p030_Exponential: Exponential Function                                          #
# 032. p032_Griewank: Griewank Function                                                #
# 037. p037_Hosaki: Hosaki Function                                                    #
# 045. p045_Mishra: Mishra Function                                                    #
# 049. p049_Paviani: Paviani Function                                                  #
# 054. p054_PowellSum: PowellSum Function                                              #
# 057. p057_Quartic: Quartic Function                                                  #
# 058. p058_Quintic: Quintic Function                                                  #
# 065. p065_Salomon: Salomon Function                                                  #
# 071. p071_SchumerSteiglitz: SchumerSteiglitz Function                                #
# 075. p075_Sphere: Sphere Function                                                    #
# 076. p076_Step: Step Function                                                        #
# 077. p077_Stepint: Stepint Function                                                  #
# 079. p079_SumSquares: SumSquares Function                                            #
# 080. p080_StyblinskiTang: Styblinski - Tang Function                                 #
# 096. p096_XinSheYang2: XinSheYangSecond Function                                     #
# 098. p098_XinSheYang4: XinSheYangFourth Function                                     #
# 100. p100_Zakharov: Zakharov Function                                                #
# ==================================================================================== #
"""
import matplotlib.pyplot as plt
from py_hybrid_function.GWO import *
from py_hybrid_function.GWO_CS import *
from py_hybrid_function.func_plot import *

# Number of search agents
SearchAgents_no = 20
# Name of the test function
Function_name = 'p001_Ackley'
# Maximum number of iterations
Max_iteration = 500

# Load details of the selected benchmark function

# GWO part:
(Best_score, Best_pos, GWO_cg_curve, ptype) = GWO(Function_name, Max_iteration, SearchAgents_no)

# GWO_CS part:
(Best_score_CS, Best_pos_CS, GWOCS_cg_curve, ptype) = GWO_CS(Function_name, Max_iteration, SearchAgents_no)

# Plot
(PlotX, PlotY, PlotZ) = func_plot(Function_name)

# Draw objective space
Algorithm = 'GWO_and_GWOCS_comparison'
Label = Function_name
filename_Figure = Algorithm + '_' + Label + '.pdf'
plt.rc('text', usetex=True)
plt.rc('font', family='Segoe UI')
'Plot the convergence history'
SF = 14 # Scale factor

fig, (ax1, ax2) = plt.subplots(1, 2)

#Plot 3D
ax1 = fig.add_subplot(121, projection = '3d')
ax1.plot_surface(PlotX, PlotY, PlotZ, cmap = 'viridis')
ax1.set_title('3D Graph', fontname = "Segoe UI", fontsize = SF, color = 'k')

# line plot of best objective function values
if ptype == 0:
    ax2.semilogy(GWO_cg_curve, label="GWO", color='b', linewidth=3)
    ax2.semilogy(GWOCS_cg_curve, label = "GWOCS", color = 'r', linestyle = '-.', linewidth=2)
else:
    ax2.plot(GWO_cg_curve, label="GWO", color='b', linewidth=3)
    ax2.plot(GWOCS_cg_curve, label = "GWOCS", color = 'r', linestyle='-.', linewidth=2)
ax2.set_title('Convergence curve', fontname = "Segoe UI", fontsize=SF, color='k')
ax2.set_xlabel('Iteration', fontname = "Segoe UI", fontsize=SF, color='k')
ax2.set_ylabel('Best score obtained so far', fontname = "Segoe UI", fontsize=SF, color='k')
leg = ax2.legend(loc = 'upper right')
plt.show()
fig.savefig(filename_Figure, dpi=1000)
# Display the calculation
print('The best solution obtained by GWO is: ', str(Best_pos))
print('The best optimal value of the objective function found by GWO is: ', str(Best_score))
print('The best solution obtained by GWOCS is: ', str(Best_pos_CS))
print('The best optimal value of the objective function found by GWOCS is: ', str(Best_score_CS))

