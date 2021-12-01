import numpy as np
def p096_XinSheYang2(x):
    dim = len( x )
    x = x.flatten()
    xsum = 0
    xsumsin = 0
    for i in range( dim ):
        xsum += abs(x[i])
        xsumsin += np.sin( (x[i]**2 ) )
    o = xsum*np.exp(-xsumsin)
    return (o)
