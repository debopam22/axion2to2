import numpy as np
from constants import *
from scipy.special import spherical_jn


class FormFactorSquared:
    def __init__(self, nameff="Dipole", t=-1.0, A=40, Z=18):
        self.nameff=nameff
        self.t=t
        self.A=A
        self.Z=Z


    def HelmFFsquared(self):
        t = np.array(self.t)
        t[t > 0] = 0
        Q = np.sqrt(-t)
        R, s = 3.9 * ((self.A / 40) ** (1 / 3)) * (1e-13) * cm, 0.9 * (1e-13) * cm
        x = Q * R
        FERHelm = np.ones_like(x)
        mask = x > 1e-6
        FERHelm[mask] = (3 * spherical_jn(1, x[mask]) * np.exp(-Q[mask] ** 2 * s ** 2 / 2) / x[mask])
        FERHelm = np.nan_to_num(FERHelm, nan=0.0)

        return np.square(np.abs(FERHelm))                   # Dimensionless


    def DipoleFFsquared(self):
        Lambdap = 0.77                                      # GeV
        term1 = self.t / (Lambdap ** 2)
        exp = (1 / (1 - term1)) ** 4

        return np.nan_to_num(exp, nan=0)                    # Dimensionless


    def atomicsquared(self):
        r0 = 0.8853*0.529*1e-8*(self.Z**(-1/3))*cm           # GeV^-1
        exp = (qe**2*self.t)/(self.t+(1/(r0**2)))

        return np.nan_to_num(exp**2, nan=0)  # Dimensionless


    def formfactor(self):
        if self.nameff.lower()=="dipole":
            return self.DipoleFFsquared()
        elif self.nameff.lower()=="helm":
            return self.HelmFFsquared()
        elif self.nameff.lower()=="atomic":
            return self.atomicsquared()
        else:
            print("Wrong Form Factor name or Form Factor not in the list or Electron Final State")
            return np.ones_like(self.t)


#print(FormFactorSquared(nameff="Helm", t=-1.0, A=40).formfactor())