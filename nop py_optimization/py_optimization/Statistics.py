"""
#==================================================================================================#
# Comparison results of DE and NMR through benchmark test Appendix A                               #
# This work has been done by:                                                                      #
# 1. Nguyen Anh Khoa - 1810240                                                                     #
# 2. Phan Vuong Phu - 1710235                                                                      #
# 3. Trang Si Tan Khang - 1810215                                                                  #
#==================================================================================================#
"""
import pandas as pd
import numpy as np

# Input analytical fx value
fx = 0

DEarray = np.array([])
NMRarray = np.array([])

# Open DE xls file
num = 0
dbest = 9999
dworst = 0
# Input link for DE
xls = pd.ExcelFile(r"C:\Users\robin\Desktop\py_optimization\DE_Survey_p100_Zakharov.xls")
sheetDE = xls.parse(0)
var1 = sheetDE['Best solution']
comp = abs(float(np.mean(var1)))
if comp < 1 : comp = 1
for i in range(1, 100):
    if abs(var1[i] - fx) <= comp:
        if dbest > abs(var1[i] - fx):
            dbest = abs(var1[i] - fx)
            fbest = var1[i]
        if dworst < abs(var1[i] - fx):
            dworst = abs(var1[i] - fx)
            fworst = var1[i]
        DEarray = np.append(DEarray, var1[i])
        num += 1
meanDE = np.mean(DEarray)
stdDE = np.std(DEarray)
# Print results
print('Results for DE Algorithm:')
print('Fbest = ', fbest)
print('Fworst = ', fworst)
print('Mean value = ', meanDE)
print('Standard Deviation = ', stdDE)

# Open NMR xls file
num = 0
dbest = 9999
dworst = 0
# Input link for NMR
xls = pd.ExcelFile(r"C:\Users\robin\Desktop\py_optimization\NMR_Survey_p100_Zakharov.xls")
sheetNMR = xls.parse(0)
var2 = sheetNMR['Best solution']
comp2 = abs(float(np.mean(var2)))
if comp2 < 1: comp2 = 1
for i in range(1, 100):
    if abs(var2[i] - fx) <= comp2:
        if dbest > abs(var2[i] - fx):
            dbest = abs(var2[i] - fx)
            fbest = var2[i]
        if dworst < abs(var2[i] - fx):
            dworst = abs(var2[i] - fx)
            fworst = var2[i]
        NMRarray = np.append(NMRarray, var1[i])
        num += 1
meanDE = np.mean(NMRarray)
stdDE = np.std(NMRarray)
# Print results
print('Results for NMR Algorithm:')
print('Fbest = ', fbest)
print('Fworst = ', fworst)
print('Mean value = ', meanDE)
print('Standard Deviation = ', stdDE)