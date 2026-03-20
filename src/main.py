import argparse
from scattering import *

parser = argparse.ArgumentParser(...)

parser.add_argument("--m1",         type=float, default=67.6, help="Mass parameter m1 for particle 1 (default: 37.2)")
parser.add_argument("--m2",  type=float, default=0.0, help="Mass parameter m2 for particle 2 (default: 0.0)")
parser.add_argument("--m3",          type=float, default=67.6, help="Mass parameter m3 for particle 3 (default: 37.2)")
parser.add_argument("--m4",     type=float, default=0.0, help="Mass parameter m4 for particle 4 (default: 0.01)")
parser.add_argument("--mi1",     type=float, default=0.0, help="Mass parameter m1 for mediator particle (default: 0.0)")
parser.add_argument("--g12i1",          type=float, default=1.0, help="Coupling between particles 1-2-mediator (default 1.0)")
parser.add_argument("--g13i1",          type=float, default=1.0, help="Coupling between particles 1-3-mediator (default 1.0)")
parser.add_argument("--g24i1",    type=float,   default=1e-8, help="Coupling between particles 2-4-mediator (default 1.0)")
parser.add_argument("--g34i1",          type=float, default=1.0, help="Coupling between particles 3-4-mediator (default 1.0)")
parser.add_argument("--nameprocess",     type=str,   default="InversePrimakoff",
                        choices=["Primakoff", "InversePrimakoff", "Compton", "InverseCompton"], help="What type of process do you need? (Primakoff, InversePrimakoff, Compton, InverseCompton)")
parser.add_argument("--nameff",     type=str,   default="Electron",
                        choices=["Dipole", "Helm", "Atomic", "Electron"], help="What is your form factor? (Dipole, Helm, Atomic, Electron)")
parser.add_argument("--goal",     type=str,   default="CrossSection",
                        choices=["CrossSection", "Particle3Spectra", "Particle4Spectra"], help="What is your goal? (CrossSection, Particle3Spectra, Particle4Spectra)")
parser.add_argument("--Nsamples_MC", type=int,   default=int(1e6), help="The total number of Monte Carlo sample required (default 10^6)?")
parser.add_argument("--A",   type=int,   default=72, help="Mass number of the nucleus if present (default Ar nucleus and A=40)")
parser.add_argument("--Z",  type=int,   default=32, help="Atomic number of the nucleus if present (default Ar nucleus and Z=18)")

args = parser.parse_args()
m1 = args.m1
m2 = args.m2
m3 = args.m3
m4 = args.m4
mi1 = args.mi1
g12i1 = args.g12i1
g13i1 = args.g13i1
g24i1 = args.g24i1
g34i1 = args.g34i1
nameprocess = args.nameprocess
nameff = args.nameff
Nsamples_MC = args.Nsamples_MC
A = args.A
Z = args.Z
goal = args.goal


if __name__ == "__main__":
    if goal.lower() == "crosssection":
        E2=1e-3
        res = General2to2Process(E2=E2, m1=m1, m2=m2, m3=m3, m4=m4, g13i1=g13i1, g24i1=g24i1, g12i1=g12i1, g34i1=g34i1,
                                  mi1=mi1, Nsamples_MC=Nsamples_MC, A=A, Z=Z, nameprocess=nameprocess,
                                  nameff=nameff).crosssection_total()
        print(f"The total cross-section in cm^2: ", res)
    elif goal.lower() == "particle3spectra":
        E2=5.0
        res = General2to2Process(E2=E2, m1=m1, m2=m2, m3=m3, m4=m4, g13i1=g13i1, g24i1=g24i1, g12i1=g12i1, g34i1=g34i1,
                                  mi1=mi1, Nsamples_MC=Nsamples_MC, A=A, Z=Z, nameprocess=nameprocess,
                                  nameff=nameff).crosssection_spectra()
        momenta = kinematics2to2(m1=m1, m2=m2, m3=m3, m4=m4, mi1=mi1, E2=E2).labframemomenta3(t=res[1])
        print(np.shape(np.transpose(momenta)))
    elif goal.lower() == "particle4spectra":
        E2=5.0
        res = General2to2Process(E2=E2, m1=m1, m2=m2, m3=m3, m4=m4, g13i1=g13i1, g24i1=g24i1, g12i1=g12i1, g34i1=g34i1,
                                  mi1=mi1, Nsamples_MC=Nsamples_MC, A=A, Z=Z, nameprocess=nameprocess,
                                  nameff=nameff).crosssection_spectra()
        kinematics2to2(m1=m1, m2=m2, m3=m3, m4=m4, mi1=mi1, E2=E2).labframemomenta4(t=res[1])
    else:
        print("Wrong goal!")
