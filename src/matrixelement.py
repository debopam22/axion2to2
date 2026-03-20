import numpy as np
from constants import *

class matrixclass:
    def __init__(self, ma, mN, Z, A, gaaa, gaee, nameprocess):
        self.ma=ma
        self.mN=mN
        self.Z=Z
        self.A=A
        self.gaaa=gaaa
        self.gaee=gaee
        self.nameprocess=nameprocess


    def primakoffscat(self, s, t):
        exp = -((self.gaaa**2 * qe**2 * (2 * self.mN**4 * t + 2 * self.mN**2 * (self.ma**4 - (self.ma**2 + 2 * s) * t) +
                                    t * (self.ma**4 + 2 * s**2 + 2 * s * t + t**2 - 2 * self.ma**2 * (s + t))) *
                 self.Z**2) / (4 * t**2))
        return np.nan_to_num(exp, nan=0.0)                      # Dimensionless


    def invprimakoffscat(self, s, t):
        exp = -((self.gaaa**2 * qe**2 * (2 * self.mN**4 * t + 2 * self.mN**2 * (self.ma**4 - (self.ma**2 + 2 * s) * t) +
                                    t * (self.ma**4 + 2 * s**2 + 2 * s * t + t**2 - 2 * self.ma**2 * (s + t))) *
                 self.Z**2) / (2 * t**2))
        return np.nan_to_num(exp, nan=0.0)                      # Dimensionless


    def comptonscat(self, s, t):
        exp = (self.gaee**2 * qe**2 * (self.ma**6 * (3 * me**2 - s) + self.ma**2 * (-16 * (me**3 - me * s)**2 + 12 * me**2 *
                                                                     (me**2 - s) * t + (3 * me**2 - s) * t**2) +
                                  self.ma**4 * (-7 * me**4 + me**2 * (6 * s - 5 * t) + s * (s + t)) +
                                  (me**2 - s) * (24 * me**6 + 4 * me**4 * (-6 * s + t) - t**2 * (s + t) -
                                                 me**2 * t * (4 * s + 3 * t)))) / ((me**2 - s)**2 * (self.ma**2 + me**2 - s - t)**2)
        return np.nan_to_num(exp, nan=0.0)                      # Dimensionless


    def invcomptonscat(self, s, t):
        exp = (2 * self.gaee**2 * qe**2 * (self.ma**6 * (3 * me**2 - s) + self.ma**2 * (3 * me**2 - s) * t**2 +
                                      (me**2 - s) * (me**2 - s - t) * t**2 + self.ma**4 * (me**4 + s * (s + t) -
                                       me**2 * (2 * s + 5 * t)))) / ((me**2 - s)**2 * (self.ma**2 + me**2 - s - t)**2)
        return np.nan_to_num(exp, nan=0.0)                      # Dimensionless


    def matrixelementfunction(self, s, t):
        if self.nameprocess.lower()=="primakoff": return self.primakoffscat(s, t)
        elif self.nameprocess.lower() == "inverseprimakoff": return self.invprimakoffscat(s, t)
        elif self.nameprocess.lower() == "compton": return self.comptonscat(s, t)
        elif self.nameprocess.lower() == "inversecompton": return self.invcomptonscat(s, t)
        else:
            print("Wrong process name")
            return None


#print(matrixclass(ma=0.01, mN=37.2, Z=18, A=40, gaaa=1.0, gaee=1.0, nameprocess="Primakoff").matrixelementfunction(s=10.0, t=-0.01))