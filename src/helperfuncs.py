import numpy as np

def boost3dparams(Eparent, mparent, pxparent, pyparent, pzparent):
    gam = Eparent / mparent
    betax = pxparent / Eparent
    betay = pyparent / Eparent
    betaz = pzparent / Eparent
    return gam, betax, betay, betaz


def lorentz_boost(Eparent, mparent, pxparent, pyparent, pzparent, Eni, pxi, pyi, pzi):
    gam, betax, betay, betaz = boost3dparams(Eparent, mparent, pxparent, pyparent, pzparent)
    betatot = np.sqrt(betax ** 2 + betay ** 2 + betaz ** 2)
    X00, X01, X02, X03 = gam, -gam * betax, -gam * betay, -gam * betaz
    X10, X11, X12, X13 = -gam * betax, 1 + (gam - 1) * (betax / betatot) ** 2, (gam - 1) * betax * betay / betatot ** 2, (gam - 1) * betax * betaz / betatot ** 2
    X20, X21, X22, X23 = -gam * betay, (gam - 1) * betay * betax / betatot ** 2, 1 + (gam - 1) * (betay / betatot) ** 2, (gam - 1) * betay * betaz / betatot ** 2
    X30, X31, X32, X33 = -gam * betaz, (gam - 1) * betaz * betax / betatot ** 2, (gam - 1) * betaz * betay / betatot ** 2, 1 + (gam - 1) * (betaz / betatot) ** 2

    Enf = X00 * Eni + X01 * pxi + X02 * pyi + X03 * pzi
    pxf = X10 * Eni + X11 * pxi + X12 * pyi + X13 * pzi
    pyf = X20 * Eni + X21 * pxi + X22 * pyi + X23 * pzi
    pzf = X30 * Eni + X31 * pxi + X32 * pyi + X33 * pzi

    return Enf, pxf, pyf, pzf



def lambda_function(x, y, z):
        exp=(x-y-z)**2-(4*y*z)
        if np.any(exp<0):
            return np.nan
        return exp

def Gfunction(x, y, z, u, v, w):
    exp=v**2*w+u**2*z+u*((w-x)*(-v+y)-(v+w+x+y)*z+z**2)+x*(y*(x+y-z)+w*(-y+z))+v*(w**2+y*(-x+z)-w*(x+y+z))
    if np.any(exp>0):
        print("G Function greater than 0")
        return np.nan
    return exp