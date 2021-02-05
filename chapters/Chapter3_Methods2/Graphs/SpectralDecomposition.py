###   SPECTRAL DECOMPOSITION   ###

import numpy as np
from scipy.optimize import least_squares

def Residuals(params, Nt, Dat, StdDat):
    Sig_v = np.array(params)
    diff = np.power((Dat - Nt@Sig_v)/StdDat,2.0)       
    return diff.reshape(-1)

def Chi2_Iso(Par, Lambda_Pop, freq, time, data, data_e):
    
    KineticModel = np.array([[ -Par[0],       0.0,   0.0],
                             [ +Par[0],   -Par[1],   0.0],
                             [     0.0,   +Par[1],   0.0]])

    EigenVectors, EigenValues = KineticModel.diagonalize()

    N0 = np.array([1,0,0])

    #Given the decay rates, compute the population matrix 
    Popt = EigenVectors@np.exp(EigenValues*time)@(EigenVectors**-1)@N0

    #Create the spectra matrix
    Spectra = np.zeros((len(N0),len(freq)), dtype = 'float64')

    #Initial zeros
    p0 = np.zeros((len(N0)),dtype='float64')    
    
    #Find the frequency components that best fit the experimental data
    for i in range(len(freq)):
        params = least_squares(Residuals, p0, args=(Popt, data[:,i], data_e[:,i]), method='trf', loss='linear', max_nfev = 1000)
        p0 = params['x']      
        Spectra[:,i] = params['x']        

    #Save the spectral signature for future use
    np.savetxt('Write_FileName',Spectra,fmt='%.18e', delimiter='\t')

    #Chi-square function
    X2 = np.power((data - Popt@Spectra)/data_e,2.0)
    return X2.reshape(-1)

#Importing experimental data       
t = np.loadtxt('File_Time')
omega = np.loadtxt('File_Frequency')

Iso_signal = np.loadtxt('File_ExperimentalData')      
Iso_error = np.loadtxt('File_ExperimentalStDev')      

#Initial estimation: decay rates
k0 = [1.0/1.7, 1.0/1.2]   

#Least-square minimization
OptPar = least_squares(Chi2_Iso, k0, args=(omega, t, Iso_signal, Iso_error), method='trf', loss='linear', max_nfev = 3000)
        
### OptPar['x']:    optimized decay rates   
### OptPar['jac']:  optimized Jacobian to be used for statistical analysis      