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
# 071. p071_Schumer: SchumerSteiglitz Function                                                     #
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
import xlwt
from py_function.DE import *

NMR_pop = 20                     # Number of NMRs
Function_name = 'p100_Zakharov'    # Name of the test function
Max_iteration = 1000           # Maximum number of iterations

# define population size
pop_size = 10
# define number of iterations
maxiter = 1000
# define scale factor for mutation
F = 0.5
# define crossover rate for recombination
cr = 0.7

# Export to Excel
workbook = xlwt.Workbook()
sheetName = "DE_Survey_" + Function_name
sheet = workbook.add_sheet(sheetName, cell_overwrite_ok=True)
sheet.write(0, 0, "Number")
sheet.write(0, 1, "Best solution")
sheet.write(0, 2, "Best optimal value")

Number_of_Cycle = 100            # Choose survey sample space
startRow = 1
for i in range(Number_of_Cycle):
    # Run DE Algorithm
    # perform differential evolution
    solution = DE(Function_name, pop_size, maxiter, F, cr)
    sheet.write(startRow, 0, str(startRow))
    sheet.write(startRow, 1, str(solution[1]))
    sheet.write(startRow, 2, str(solution[0]))
    startRow += 1
xlsName = sheetName + ".xls"
workbook.save(xlsName)
