"""
%====================================================================================%
% Hybrid Differential Evolution Algorithm With Adaptive Crossover Mechanism - DE_TCR %
%                                                                                    %
% Student: Trang Si Tan Khang                                                        %
% Code: 1810215                                                                      %
%                                                                                    %
% Basic features of the algorithm are:                                               %
% 1. It uses a population management mechanims to improve exploration                %
% 2. It has an adaptive mechanism for crossover rate                                 %
% 3. It uses a local search routine to improve convergence                           %
%                                                                                    %
% Scripts and functions listing:                                                     %
% Main.py              - Main script to run the algorithm                            %
% DE_TCR.py            - The optimization algorithm                                  %
% DE_TCRparam.py       - Script to build the struct required for the optimization    %
% LocalSearch.py       - The Local Search subroutine                                 %
% CostFunction.py      - Cost function definition                                    %
%====================================================================================%
"""
import time
import matplotlib.pyplot as plt
from py_hybrid_de_function.PSOGWO import *
from py_hybrid_de_function.GWO import *
from py_hybrid_de_function.func_plot import *

SearchAgents_no = 30           # Number of search agents
Function_name = 'F18'          # Name of the test function
Max_iteration = 500            # Maximum number of iterations

# PSOGWO part:
start_time = time.time()
(Best_score, Best_pos, PSOGWO_cg_curve, ptype) = PSOGWO(SearchAgents_no, Max_iteration, Function_name)
print("Executation time for PSOGWO:")
print("--- %s seconds ---" % (time.time() - start_time))

# GWO part:
start_time = time.time()
(Alpha_score, Alpha_pos, GWO_cg_curve, ptype) = GWO(SearchAgents_no, Max_iteration, Function_name)
print("Executation time for GWO:")
print("--- %s seconds ---" % (time.time() - start_time))

# Display the calculation
print('The best solution obtained by PSOGWO is: ', str(Best_pos))
print('The best optimal value of the objective function found by PSOGWO is: ', str(Best_score))
print('The best solution obtained by GWO is: ', str(Alpha_pos))
print('The best optimal value of the objective function found by GWO is: ', str(Alpha_score))

# Plot
#(PlotX, PlotY, PlotZ) = func_plot(Function_name)

# Draw objective space
Algorithm = 'PSO_and_PSOGWO_comparison'
Label = Function_name
filename_Figure = Algorithm + '_' + Label + '.pdf'
plt.rc('text', usetex=True)
plt.rc('font', family='Segoe UI')
'Plot the convergence history'
SF = 14 # Scale factor

#fig, (ax1, ax2) = plt.subplots(1, 2)
fig = plt.figure()   #figsize=plt.figaspect(2.))
fig.suptitle('Comparison of PSOGWO and GWO')

#Plot 3D
#ax = fig.add_subplot(121, projection = '3d')
#surf = ax.plot_surface(PlotX, PlotY, PlotZ, cmap = 'viridis', rstride=1, cstride=1, linewidth=0, antialiased=False)
#ax.set_title('3D Graph', fontname = "Segoe UI", fontsize = SF, color = 'k')

#fig.tight_layout(pad=4.0)

# line plot of best objective function values
#ax = fig.add_subplot(122)
ax = fig.add_subplot()
if ptype == 0:
    ax.semilogy(GWO_cg_curve, label = "GWO", color = 'b', linewidth = 3)
    ax.semilogy(PSOGWO_cg_curve, label = "PSOGWO", color = 'r', linestyle = '-.', linewidth = 2)
else:
    ax.plot(GWO_cg_curve, label = "GWO", color = 'b', linewidth = 3)
    ax.plot(PSOGWO_cg_curve, label = "PSOGWO", color = 'r', linestyle = '-.', linewidth = 2)
ax.set_title('Convergence curve', fontname = "Segoe UI", fontsize = SF, color = 'k')
ax.set_xlabel('Iteration', fontname = "Segoe UI", fontsize = SF, color = 'k')
ax.set_ylabel('Best score obtained so far', fontname = "Segoe UI", fontsize = SF, color = 'k')
leg = ax.legend(loc = 'upper right')
plt.show()
fig.savefig(filename_Figure, dpi = 1000)



