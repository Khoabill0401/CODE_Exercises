import numpy as np

def p011_Brown(x):
    dim = len (x)
    x = x.flatten ()
    sum = 0
    for i in range (dim -1):
        a = (x[i]**2)**(x[1+1]**2+ 1)
        b = (x[i+1]**2)**(x[i]**2 +1)
        sum = sum + a + b
        return (sum)