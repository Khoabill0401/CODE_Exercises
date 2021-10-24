import numpy as np
def p096_XinSheYangSecond(x):
    dim = len( x )
    x = x.flatten()
    xsum = 0
    xsumsin = 0
    for i in range( dim ):
        xsum += abs(x[i])
        xsumsin += np.sin( np.pi * (x[i]**2 ) )
    o = xsum*np.exp(-xsumsin)
    return (o)
