
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import leastsq

eps0 = 8.8541878176e-12 #Permittivity of free space

#Cole-Cole relaxation mode plus ionic conductivity term
def ColeCon(x, AD, tau, alfa, kappa):
    CC = 5.915 + (AD)/(1 + ((1j*(2*np.pi)*(tau)*(x))**(1 - alfa)))
    Con = (kappa)/(2*np.pi*(eps0)*(x))*1j
    return CC - Con

#Residual function
def Residual(P,f,epsR,epsI):
    ResR = epsR - ColeCon(f, *P).real
    ResI = epsI - ColeCon(f, *P).imag
    return np.sqrt(np.power(ResR, 2) + np.power(ResI, 2))

#Importing raw permittivity
Freq, epsRaw_Real, epsRaw_Imag = np.loadtxt('File_Name', unpack=True)

#Initial values
p0=[65.0, 8.7e-12, 0.01, 7.0]

#Least-square minimization
ParOpt, CovOpt, info, errmsg, ier = leastsq(Residual, p0, args=(Freq, epsRaw_Real, epsRaw_Imag), full_output=1)

### ParOpt: optimized parameters
### CovOpt: optimized covariance matrix