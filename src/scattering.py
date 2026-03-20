# Contains the kinematics of the 2-->2 Scattering

from formfactors import FormFactorSquared
from matrixelement import matrixclass
from constants import *
from helperfuncs import *

np.random.seed(42)

class General2to2Process:
    """
    Particle 1 is at rest and 2 is coming with some energy in the lab frame.
    Particles 3 and 4 are moving in the lab frame.
    mi1 is the mass of the mediator
    gijmi1 is the coupling between i-j-i1 particles
    A is the mass number of the nucleus if nucleus is present in the 2-to-2 scattering. Generally if nucleus or electron
    is present then they are considered to be at rest (particle 1) in the lab frame. Although this is not mandatory.
    Nsamples_MC is the number of Monte Carlo samples we generate.
    E2 is the energy of the incoming particle in the lab frame.
    Z is the atomic number of the nucleus if present in the 2-->2 scattering.
    nameprocess is the name of the 2-->2 process to be considered.
    nameff is the name of the form factor to be considered.
    """
    def __init__(self, E2, m1, m2, m3, m4, g13i1, g24i1, g12i1, g34i1, mi1, Nsamples_MC, A, Z, nameprocess, nameff):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.E2 = E2
        self.s = self.m1 ** 2 + self.m2 ** 2 + 2 * self.E2 * self.m1
        self.g13i1 = g13i1
        self.g24i1 = g24i1
        self.g12i1 = g12i1
        self.g34i1 = g34i1
        self.Nsamples_MC = Nsamples_MC
        self.mi1 = mi1
        self.A = A
        self.Z=Z
        self.nameprocess=nameprocess
        self.nameff=nameff

    def E1cm(self):
        E1 = (self.s + self.m1 ** 2 - self.m2 ** 2) / (2 * np.sqrt(self.s))
        return np.where(E1 >= self.m1, E1, 0)

    def E2cm(self):
        E2 = (self.s - self.m1 ** 2 + self.m2 ** 2) / (2 * np.sqrt(self.s))
        return np.where(E2 >= self.m2, E2, 0)

    def E3cm(self):
        E3 = (self.s + self.m3 ** 2 - self.m4 ** 2) / (2 * np.sqrt(self.s))
        return np.where(E3 >= self.m3, E3, 0)

    def E4cm(self):
        E4 = (self.s - self.m3 ** 2 + self.m4 ** 2) / (2 * np.sqrt(self.s))
        return np.where(E4 >= self.m4, E4, 0)

    def p1cm(self):
        p1 = self.E1cm() ** 2 - self.m1 ** 2
        return np.where(p1 >= 0, np.sqrt(p1), 0)

    def p2cm(self):
        p2 = self.E2cm() ** 2 - self.m2 ** 2
        return np.where(p2 >= 0, np.sqrt(p2), 0)

    def p3cm(self):
        p3 = self.E3cm() ** 2 - self.m3 ** 2
        return np.where(p3 >= 0, np.sqrt(p3), 0)

    def p4cm(self):
        p4 = self.E4cm() ** 2 - self.m4 ** 2
        return np.where(p4 >= 0, np.sqrt(p4), 0)

    def tlowerlim(self):
        exp = self.m1 ** 2 + self.m3 ** 2 - (1 / (2 * self.s)) * ((self.s + self.m1 ** 2 - self.m2 ** 2) * (self.s + self.m3 ** 2 - self.m4 ** 2) + np.sqrt(lambda_function(self.s, self.m1 ** 2, self.m2 ** 2) * lambda_function(self.s, self.m3 ** 2, self.m4 ** 2)))
        return np.nan_to_num(exp, nan=0)

    def tupperlim(self):
        exp = self.m1 ** 2 + self.m3 ** 2 - (1 / (2 * self.s)) * ((self.s + self.m1 ** 2 - self.m2 ** 2) * (self.s + self.m3 ** 2 - self.m4 ** 2) - np.sqrt(lambda_function(self.s, self.m1 ** 2, self.m2 ** 2) * lambda_function(self.s, self.m3 ** 2, self.m4 ** 2)))
        return np.nan_to_num(exp, nan=0)

    def dsigmadt(self, t):
        return np.nan_to_num((1 / (16 * np.pi * (self.s - (self.m1 + self.m2) ** 2) * (self.s - (self.m1 - self.m2) ** 2))) * matrixclass(ma=self.mi1, mN=self.m1, Z=self.Z, A=self.A, gaaa=self.g24i1, gaee=self.g13i1, nameprocess=self.nameprocess).matrixelementfunction(s=self.s, t=t), nan=0.0) #avgmatrixsquared2(t, self.g13i1, self.g24i1, self.m1, self.m2, self.m3, self.mi1), nan=0)

    def dsigmadEr(self, Er):
        return np.nan_to_num((1/(16*np.pi*(self.s-(self.m1+self.m2)**2)*(self.s-(self.m1-self.m2)**2)))*np.abs((-2*self.m2))*matrixclass(ma=self.mi1, mN=self.m1, Z=self.Z, A=self.A, gaaa=self.g24i1, gaee=self.g13i1, nameprocess=self.nameprocess).matrixelementfunction(s=self.s, t=-2*self.m1*Er), nan=0)

    def crosssection_spectra(self):
        tlow, tup = self.tlowerlim(), self.tupperlim()
        t_range = np.random.uniform(tlow, tup, self.Nsamples_MC)                              #uniform sampling
        #tup_log_abs, tlow_log_abs = np.log10(np.abs(tlow)), np.log10(np.abs(tup))
        #t_range = -10 ** (np.random.uniform(tlow_log_abs, tup_log_abs, self.Nsamples_MC))      #uniform sampling in log space

        condition = (t_range < 0) & (self.s>=max((self.m1+self.m2)**2,(self.m3+self.m4)**2))
        t_range = t_range[condition]
        if len(t_range) == 0:
            return np.array([0]), np.array([0])

        int_result = self.dsigmadt(t_range)
        volume = (max(t_range) - min(t_range))

        Nsampremain = self.Nsamples_MC - np.sum(np.where(condition, 0, 1))
        #print("Percentage of samples remaining (%): ", (Nsampremain*100/self.N), Nsampremain)
        integral = (int_result * FormFactorSquared(nameff=self.nameff, t=t_range, A=self.A, Z=self.Z).formfactor() * volume) * (1 / (cm ** 2 * Nsampremain))
        return integral, t_range        #cm^2, GeV^2


    def crosssection_total(self):
        tlow, tup = self.tlowerlim(), self.tupperlim()
        t_range = np.random.uniform(tlow, tup, self.Nsamples_MC)                                  #uniform sampling
        #tup_log_abs, tlow_log_abs = np.log10(np.abs(tlow)), np.log10(np.abs(tup))
        #t_range = -10 ** (np.random.uniform(tlow_log_abs, tup_log_abs, self.Nsamples_MC))          #uniform sampling in log space

        condition = (t_range < 0) & (self.s>=max((self.m1+self.m2)**2,(self.m3+self.m4)**2))
        t_range = t_range[condition]
        if len(t_range) == 0:
            return 0.0

        int_result = self.dsigmadt(t_range)
        volume = (max(t_range) - min(t_range))
        integral = (np.mean(int_result * FormFactorSquared(nameff=self.nameff, t=t_range, A=self.A, Z=self.Z).formfactor() * volume))*(1/(cm**2))
        return np.nan_to_num(integral, nan=0.0)                              #cm^2



class kinematics2to2:
    """
    Particle 1 is at rest and 2 is coming with some energy in the lab frame.
    Particles 3 and 4 are moving in the lab frame.
    mi1 is the mass of the mediator
    E2 is the energy of the incoming particle in the lab frame.
    """
    def __init__(self, m1, m2, m3, m4, mi1, E2):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.mi1 = mi1
        self.E2 = E2
        self.s = (self.m1 ** 2) + (self.m2 ** 2) + 2 * self.m1 * self.E2

    def E1cm(self):
        E1 = (self.s + self.m1 ** 2 - self.m2 ** 2) / (2 * np.sqrt(self.s))
        return np.where(E1 >= self.m1, E1, 0)

    def E2cm(self):
        E2 = (self.s - self.m1 ** 2 + self.m2 ** 2) / (2 * np.sqrt(self.s))
        return np.where(E2 >= self.m2, E2, 0)

    def E3cm(self):
        E3 = (self.s + self.m3 ** 2 - self.m4 ** 2) / (2 * np.sqrt(self.s))
        return np.where(E3 >= self.m3, E3, 0)

    def E4cm(self):
        E4 = (self.s - self.m3 ** 2 + self.m4 ** 2) / (2 * np.sqrt(self.s))
        return np.where(E4 >= self.m4, E4, 0)

    def p1cm(self):
        p1 = self.E1cm() ** 2 - self.m1 ** 2
        return np.where(p1 >= 0, np.sqrt(p1), 0)

    def p2cm(self):
        p2 = self.E2cm() ** 2 - self.m2 ** 2
        return np.where(p2 >= 0, np.sqrt(p2), 0)

    def p3cm(self):
        p3 = self.E3cm() ** 2 - self.m3 ** 2
        return np.where(p3 >= 0, np.sqrt(p3), 0)

    def p4cm(self):
        p4 = self.E4cm() ** 2 - self.m4 ** 2
        return np.where(p4 >= 0, np.sqrt(p4), 0)

    def costheta14CM(self, t):
        return -(t - self.m1 ** 2 - self.m3 ** 2 + 2 * self.E1cm() * self.E3cm()) / (2 * self.p1cm() * self.p3cm())

    def costheta13CM(self, t):
        return (t - self.m1 ** 2 - self.m3 ** 2 + 2 * self.E1cm() * self.E3cm()) / (2 * self.p1cm() * self.p3cm())

    # For 2-->2 scattering, the boosting is done with a +ve sign for the parent particle's momentum (which is at rest in the lab frame, particle 1)
    def labframemomenta3(self, t):
        theta_m1, phi_m1=np.arccos(1 - 2 * np.random.rand(len(t))), 2 * np.pi * np.random.rand(len(t))
        px_m1, py_m1, pz_m1=self.p1cm()*np.sin(theta_m1)*np.cos(phi_m1), self.p1cm()*np.sin(theta_m1)*np.sin(phi_m1), self.p1cm()*np.cos(theta_m1)
        costheta = self.costheta13CM(t)
        try:
            phi2 = 2 * np.pi * np.random.rand(len(costheta))
        except:
            phi2 = 2 * np.pi * np.random.rand()
        four_vectors = lorentz_boost(Eparent=self.E1cm(), mparent=self.m1, pxparent=px_m1,
                                          pyparent=py_m1, pzparent=pz_m1, Eni=self.E3cm(),
                                          pxi=self.p3cm() * np.sqrt(1 - costheta ** 2) * np.cos(phi2),
                                          pyi=self.p3cm() * np.sqrt(1 - costheta ** 2) * np.sin(phi2),
                                          pzi=self.p3cm() * costheta)
        energies, px, py, pz = four_vectors
        return energies, px, py, pz


    def labframemomenta4(self, t):
        theta_m1, phi_m1=np.arccos(1 - 2 * np.random.rand(len(t))), 2 * np.pi * np.random.rand(len(t))
        px_m1, py_m1, pz_m1=self.p1cm()*np.sin(theta_m1)*np.cos(phi_m1), self.p1cm()*np.sin(theta_m1)*np.sin(phi_m1), self.p1cm()*np.cos(theta_m1)
        costheta = self.costheta14CM(t)
        try:
            phi2 = 2 * np.pi * np.random.rand(len(costheta))
        except:
            phi2 = 2 * np.pi * np.random.rand()
        four_vectors = lorentz_boost(Eparent=self.E1cm(), mparent=self.m1, pxparent=px_m1,
                                          pyparent=py_m1, pzparent=pz_m1, Eni=self.E4cm(),
                                          pxi=self.p4cm() * np.sqrt(1 - costheta ** 2) * np.cos(phi2),
                                          pyi=self.p4cm() * np.sqrt(1 - costheta ** 2) * np.sin(phi2),
                                          pzi=self.p4cm() * costheta)
        energies, px, py, pz = four_vectors
        return energies, px, py, pz

#res=General2to2Process(E2=10.0, m1=37.2, m2=0, m3=37.2, m4=0.01, g13i1=1.0, g24i1=1.0, g12i1=1.0, g34i1=1.0, mi1=0.0,
#                         Nsamples_MC=100000, A=40, Z=18, nameprocess="Primakoff", nameff="Dipole").crosssection_spectra()
#print(np.shape(res))
#res=kinematics2to2(m1=37.2, m2=0, m3=37.2, m4=0.01, mi1=0.0, E2=10.0).labframemomenta3(t=res[1])
#print(np.shape(res))

