# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 09:29:50 2017

@author: mhayman
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.io import netcdf
import LidarProfileFunctions as lp
import WVProfileFunctions as wv
import FourierOpticsLib as FO
import SpectrumLib as rb

from scipy.special import wofz
import os

"""
Input Evaluation Variables:
T - Temperature in K
P - Atmospheric Pressure in Pa
B - Backscatter ratio at 780 nm
nWV - water vapor density
"""





"""
Specific state parameters
T, P, nWV

Additional inputs
dr
sim_nu
wavelength
spec_file - spectroscopy file
freq_lim = np.array([lp.c/770e-9,lp.c/768e-9])
"""

mO2 = 31.998
fo2 = 0.21103240956

def Num_Gradient(func,x0,step_size=1e-3):
    Gradient = np.zeros(x0.size)
    for ai in range(x0.size):
        xu = x0.copy()
        xl = x0.copy()
        if x0[ai] != 0:
            xu[ai] = x0[ai]*(1+step_size)            
            xl[ai] = x0[ai]*(1-step_size)
#            Gradient[ai] = (func(xu)-func(xl))/(2*step_size)
        else:
            xu[ai] = step_size
            xl[ai] = -step_size
#            Gradient[ai] = (func(step_size)-func(-step_size))/(2*step_size)
        
        Gradient[ai] = (func(xu)-func(xl))/(xu[ai]-xl[ai])
    return Gradient

def Num_Jacob(func,x0,Nout=0,step_size=1e-3):
    if Nout == 0:
        Jacobian = np.zeros((func(x0).size,x0.size))
    else:
        Jacobian = np.zeros((Nout,x0.size))
    for ai in range(x0.size):
        xu = x0.copy()
        xl = x0.copy()
        if x0[ai] != 0:
            xu[ai] = x0[ai]*(1+step_size)            
            xl[ai] = x0[ai]*(1-step_size)
#            Gradient[ai] = (func(xu)-func(xl))/(2*step_size)
        else:
            xu[ai] = step_size
            xl[ai] = -step_size
#            Gradient[ai] = (func(step_size)-func(-step_size))/(2*step_size)
        
        Jacobian[:,ai] = (func(xu)-func(xl)).flatten()/(xu[ai]-xl[ai])
    return Jacobian

def OxygenTransmission(T,P,n_wv,wavelength,dr,freq_lim=np.array([lp.c/770e-9,lp.c/768e-9]),sim_nu=np.array([]),spec_file=''):
    """
    Compute the one way frequency resolved transmission of oxygen
    """
    # fraction of O2 by number density
    fO2 = (32*0.2320+28.02*0.7547+44.01*0.00046+39.94*0.0128+20.18*0.000012+4.0*0.0000007+83.8*0.000003+131.29*0.00004)*0.2320/32.0
    
    if len(spec_file) == 0:
        spec_file = '/Users/mhayman/Documents/DIAL/O2_HITRAN2012_760_781.txt'
    
    if sim_nu.size==0:
        sim_nu = np.arange(-3e9,3e9,20e6)
    
#    inu0 = np.argmin(np.abs(sim_nu)) # index to center of frequency array
    
    n_o2=fO2*(P/(lp.kB*T)-n_wv)  # to convert atm to Pa use *101325
    ext_o2 = rb.ExtinctionFromHITRAN(lp.c/wavelength+sim_nu,T,P,(mO2*1e-3)/lp.N_A,nuLim=freq_lim,freqnorm=True,filename=spec_file).T
    T_o2 = np.exp(-np.cumsum(n_o2[np.newaxis,:]*ext_o2,axis=1)*dr)
    
    return T_o2,sim_nu
    
#def OxygenProfileRatio(T,P,n_wv,B,wavelength_on,wavelength_off,Filter,dr,sim_nu=np.array([]),GainRatio=1.0):
#    
#    if sim_nu.size==0:
#        sim_nu = np.arange(-3e9,3e9,20e6)
#    inu0 = np.argmin(np.abs(sim_nu)) # index to center of frequency array
#    
#    Badj = (wavelength_on/780.24e-9)**3*(B-1)+1  # backscatter ratio adjusted for wavelength difference
#    
#    # calculate Rayleigh Brioullion scattering 
#    mol_RB = lp.RB_Spectrum(T,P,wavelength_on,nu=sim_nu,norm=True)
#    
#    T_on,_ = OxygenTransmission(T,P,n_wv,wavelength_on,dr,sim_nu=sim_nu)
#    Tf_on = Filter.spectrum(sim_nu+lp.c/wavelength_on,InWavelength=False,aoi=0.0,transmit=True)     
#    eta_on = np.sum(T_on*Tf_on[:,np.newaxis]*mol_RB,axis=0)/(T_on[inu0,:]*Tf_on[inu0])
#    
#    T_off,_ = OxygenTransmission(T,P,n_wv,wavelength_off,dr,sim_nu=sim_nu)
#    Tf_off = Filter.spectrum(sim_nu+lp.c/wavelength_off,InWavelength=False,aoi=0.0,transmit=True)     
#    eta_off = np.sum(T_off*Tf_off[:,np.newaxis]*mol_RB,axis=0)/(T_on[inu0,:]*Tf_off[inu0])
#    
#    ChannelRatio = GainRatio*T_on[inu0,:]/T_off[inu0,:]*(Badj+eta_on-1)/(Badj+eta_off-1)
#    
#    return ChannelRatio
    
def WaterVaporTransmission(T,P,n_wv,wavelength,dr,freq_lim=np.array([lp.c/828.5e-9,lp.c/828e-9]),sim_nu=np.array([])):
    """
    Compute the one way frequency resolved transmission of water vapor
    """
    
    if sim_nu.size==0:
        sim_nu = np.arange(-3e9,3e9,20e6)
    
    ext_wv = rb.ExtinctionFromHITRAN(lp.c/wavelength+sim_nu,T,P,(lp.mH2O*1e-3)/lp.N_A,nuLim=freq_lim,freqnorm=True).T
    T_wv = np.exp(-np.cumsum(n_wv[np.newaxis,:]*ext_wv,axis=1)*dr)
    
    return T_wv,sim_nu

def TDRatios(Prof,x,P,Trx,rb_spec,abs_spec,dr,inu0,bsrMult):
    """
    Prof - dict of ratio profiles to be reconstructed
    x - optimzation variable containing T,nWV,BSR as well as gain coefficients
    P - pressure profile in atm
    Trx - dict containing all the receiver transmission functions
    rb_spec - dict containing all the RB PCA data for all the channels
    abs_spec - dict containing all the absoprtion PCA data for DIAL channels
    dr - range bin size
    inu0 - dict containing the index to the center frequency of each channel
    bsrMult - dict containing the BSR multiplier due to wavelength difference
    
    dict entries:
        'HSRL Mol'
        'HSRL Comb'
        'WV Online'
        'WV Offline'
        'O2 Online'
        'O2 Offline'
    ratio entries:
        'HSRL'
        'WV'
        'O2'
    """
    
    iR = Prof['WV'].size # range index for a profile into 1D x array
    x2 = np.reshape(x,(iR+1,3))
    xK = x2[0,:]    # constants [HSRL,WV,O2]
    xS = x2[1:,:]  # state vector [T,nWV,BSR]
    
    HSRLModel = HSRLProfileRatio(xS[:,0],P,xS[:,2], \
        Trx['HSRL Mol'],Trx['HSRL Comb'], \
        rb_spec['HSRL'],inu0['HSRL'],GainRatio=xK[0])

    WVModel = WaterVaporProfileRatio(xS[:,0],P,xS[:,1],xS[:,2]*bsrMult['WV'],
        Trx['WV Online'], Trx['WV Offline'], \
        rb_spec['WV Online'],rb_spec['WV Offline'], \
        abs_spec['WV Online'],abs_spec['WV Offline'],dr, \
        inu0['WV Online'],inu0['WV Offline'],GainRatio=xK[1])
        
    O2Model = OxygenProfileRatio(xS[:,0],P,xS[:,1],xS[:,2]*bsrMult['O2'],
        Trx['O2 Online'], Trx['O2 Offline'], \
        rb_spec['O2 Online'],rb_spec['O2 Offline'], \
        abs_spec['O2 Online'],abs_spec['O2 Offline'],dr, \
        inu0['O2 Online'],inu0['O2 Offline'],GainRatio=xK[2])
        
    return HSRLModel,WVModel,O2Model

def TDErrorFunction(Prof,ProfVar,x,P,Trx,rb_spec,abs_spec,dr,inu0,bsrMult):
    """
    Prof - dict of ratio profiles to be reconstructed
    ProfVar - dict of variances for the profiles
    x - optimzation variable containing T,nWV,BSR as well as gain coefficients
    Trx - dict containing all the receiver transmission functions
    rb_spec - dict containing all the RB PCA data for all the channels
    abs_spec - dict containing all the absoprtion PCA data for DIAL channels
    dr - range bin size
    inu0 - dict containing the index to the center frequency of each channel
    bsrMult - dict containing the BSR multiplier due to wavelength difference
    
    dict entries:
        'HSRL Mol'
        'HSRL Comb'
        'WV Online'
        'WV Offline'
        'O2 Online'
        'O2 Offline'
    ratio entrices:
        'HSRL'
        'WV'
        'O2'
    """
    
    iR = Prof['WV'].size # range index for a profile into 1D x array
    x2 = np.reshape(x,(iR+1,3))
    xK = x2[0,:]    # constants [HSRL,WV,O2]
    xS = x2[1:,:]  # state vector [T,nWV,BSR]
    
    HSRLModel = HSRLProfileRatio(xS[:,0],P,xS[:,2], \
        Trx['HSRL Mol'],Trx['HSRL Comb'], \
        rb_spec['HSRL'],inu0['HSRL'],GainRatio=xK[0])

    WVModel = WaterVaporProfileRatio(xS[:,0],P,xS[:,1],xS[:,2]*bsrMult['WV'],
        Trx['WV Online'], Trx['WV Offline'], \
        rb_spec['WV Online'],rb_spec['WV Offline'], \
        abs_spec['WV Online'],abs_spec['WV Offline'],dr, \
        inu0['WV Online'],inu0['WV Offline'],GainRatio=xK[1])
        
    O2Model = OxygenProfileRatio(xS[:,0],P,xS[:,1],xS[:,2]*bsrMult['O2'],
        Trx['O2 Online'], Trx['O2 Offline'], \
        rb_spec['O2 Online'],rb_spec['O2 Offline'], \
        abs_spec['O2 Online'],abs_spec['O2 Offline'],dr, \
        inu0['O2 Online'],inu0['O2 Offline'],GainRatio=xK[2])
        
    OptError = np.nansum((HSRLModel-Prof['HSRL'])**2/ProfVar['HSRL']) \
        +np.nansum((WVModel-Prof['WV'])**2/ProfVar['WV']) \
        +np.nansum((O2Model-Prof['O2'])**2/ProfVar['O2'])
        
    return OptError

def TDGradientFunction(Prof,ProfVar,x,P,Trx,rb_spec,abs_spec,dr,inu0,bsrMult):
    """
    Prof - dict of ratio profiles to be reconstructed
    ProfVar - dict of variances for the profiles
    x - optimzation variable containing T,nWV,BSR as well as gain coefficients
    Trx - dict containing all the receiver transmission functions
    rb_spec - dict containing all the RB PCA data for all the channels
    abs_spec - dict containing all the absoprtion PCA data for DIAL channels
    dr - range bin size
    inu0 - dict containing the index to the center frequency of each channel
    bsrMult - dict containing the BSR multiplier due to wavelength difference
    
    dict entries:
        'HSRL Mol'
        'HSRL Comb'
        'WV Online'
        'WV Offline'
        'O2 Online'
        'O2 Offline'
    ratio entries:
        'HSRL'
        'WV'
        'O2'
    """ 
    
    iR = Prof['WV'].size # range index for a profile into 1D x array
    x2 = np.reshape(x,(iR+1,3))
    xK = x2[0,:]    # constants [HSRL,WV,O2]
    xS = x2[1:,:]  # state vector [T,nWV,BSR]
    
    grad2 = np.zeros(x2.shape)   
    
    HSRLModel,dHSdB,dHSdT = HSRLProfileRatioDeriv(xS[:,0],P,xS[:,2], \
        Trx['HSRL Mol'],Trx['HSRL Comb'], \
        rb_spec['HSRL'],inu0['HSRL'],GainRatio=xK[0])

    WVModel,dWVdB,dWVdnWV,dWVdT = WaterVaporProfileRatioDeriv(xS[:,0],P,xS[:,1],xS[:,2]*bsrMult['WV'],
        Trx['WV Online'], Trx['WV Offline'], \
        rb_spec['WV Online'],rb_spec['WV Offline'], \
        abs_spec['WV Online'],abs_spec['WV Offline'],dr, \
        inu0['WV Online'],inu0['WV Offline'],GainRatio=xK[1])
        
    O2Model,dO2dB,dO2dnWV,dO2dT = OxygenProfileRatioDeriv(xS[:,0],P,xS[:,1],xS[:,2]*bsrMult['O2'],
        Trx['O2 Online'], Trx['O2 Offline'], \
        rb_spec['O2 Online'],rb_spec['O2 Offline'], \
        abs_spec['O2 Online'],abs_spec['O2 Offline'],dr, \
        inu0['O2 Online'],inu0['O2 Offline'],GainRatio=xK[2])
        
    HSRLbase = 2*(HSRLModel-Prof['HSRL'])/ProfVar['HSRL']
    WVbase = 2*(WVModel-Prof['WV'])/ProfVar['WV']
    O2base = 2*(O2Model-Prof['O2'])/ProfVar['O2']
    
    # temperature gradient
    grad2[1:,0] = np.nansum(HSRLbase[np.newaxis]*dHSdT,axis=1) \
        + np.nansum(WVbase[np.newaxis]*dWVdT,axis=1) \
        + np.nansum(O2base[np.newaxis]*dO2dT,axis=1)
    
    # water vapor gradient
    grad2[1:,1] = np.nansum(WVbase[np.newaxis]*dWVdnWV,axis=1) \
        + np.nansum(O2base[np.newaxis]*dO2dnWV,axis=1)
    
    # backscatter gradient
    grad2[1:,2] = np.nansum(HSRLbase[np.newaxis]*dHSdB,axis=1) \
        + np.nansum(WVbase[np.newaxis]*dWVdB*bsrMult['WV'],axis=1) \
        + np.nansum(O2base[np.newaxis]*dO2dB*bsrMult['O2'],axis=1)

    grad2[0,0] = np.nansum(HSRLbase*HSRLModel/xK[0])
    grad2[0,1] = np.nansum(WVbase*WVModel/xK[1])
    grad2[0,2] = np.nansum(O2base*O2Model/xK[2])
    
#    OptError = np.nansum(2*(HSRLModel-Prof['HSRL'])/ProfVar['HSRL']*) \
#        +np.nansum((WVModel-Prof['WV'])**2/ProfVar['WV']) \
#        +np.sum((O2Model-Prof['O2'])**2/ProfVar['O2'])
        
    return grad2.flatten()

def WaterVaporProfileRatioDeriv(T,P,nWV,BSR,Trx_on,Trx_off,rb_on,rb_off,abs_on,abs_off,dr,inu0_on,inu0_off,GainRatio=1.0):
    """
    Ratio is Online/Offline channels
    GainRatio: Online/Offline Gain
    Inputs:
        T - Temperature in Kelvin
        P - Pressure in atm
        nWV - water vapor density
        BSR - aerosol backscatter ratio
        Trx_on - Receiver spectrum in the online channel
        Trx_off - Receiver spectrum in the offline channel
        rb_on - dictionary of Rayleigh-Brillioun PCA terms for online wavelength
        rb_off - dictionary of Rayleigh-Brillioun PCA terms for offline wavlength
        abs_on - dictionary of Water Vapor spectrum PCA terms for online wavelength
        abs_off - dictionary of Water Vapor PCA terms for offline wavlength
        dr - range grid spacing in meters
        inu0 - index to center/transmit frequency on frequency grid
        Gain Ratio - Combined/Molecular relative gain
        
    Returns:
        ChannelRatio - the forward modeled ratio of the two channels
        dCRdB - the forward modeled derivative with backscatter ratio
        dCRdnWV - the forward modeled derivative with water vapor
        dCRdT - the forward modeled derivative with temperature
    """
    
    Nn,dNndB,dNndnWV,dNndT = WVDIALDerivative(T,P,nWV,BSR,rb_on,abs_on,Trx_on,inu0_on,GainRatio,dr)
    Nf,dNfdB,dNfdnWV,dNfdT = WVDIALDerivative(T,P,nWV,BSR,rb_off,abs_off,Trx_off,inu0_off,1.0,dr)
        
    ChannelRatio = Nn/Nf
    dCRdB = (Nf[np.newaxis]*dNndB-Nn[np.newaxis]*dNfdB)/(Nf[np.newaxis])**2
    dCRdT = (Nf[np.newaxis]*dNndT-Nn[np.newaxis]*dNfdT)/(Nf[np.newaxis])**2
    dCRdnWV = (Nf[np.newaxis]*dNndnWV-Nn[np.newaxis]*dNfdnWV)/(Nf[np.newaxis])**2
    
    
    return ChannelRatio,dCRdB,dCRdnWV,dCRdT
    
def OxygenProfileRatioDeriv(T,P,nWV,BSR,Trx_on,Trx_off,rb_on,rb_off,abs_on,abs_off,dr,inu0_on,inu0_off,GainRatio=1.0):
    """
    Ratio is Online/Offline channels
    GainRatio: Online/Offline Gain
    Inputs:
        T - Temperature in Kelvin
        P - Pressure in atm
        nWV - water vapor density
        BSR - aerosol backscatter ratio
        Trx_on - Receiver spectrum in the online channel
        Trx_off - Receiver spectrum in the offline channel
        rb_on - dictionary of Rayleigh-Brillioun PCA terms for online wavelength
        rb_off - dictionary of Rayleigh-Brillioun PCA terms for offline wavlength
        abs_on - dictionary of Oxygen spectrum PCA terms for online wavelength
        abs_off - dictionary of Oxygen PCA terms for offline wavlength
        dr - range grid spacing in meters
        inu0 - index to center/transmit frequency on frequency grid
        Gain Ratio - Combined/Molecular relative gain
        
    Returns:
        ChannelRatio - the forward modeled ratio of the two channels
        dCRdB - the forward modeled derivative with backscatter ratio
        dCRdnWV - the forward modeled derivative with water vapor
        dCRdT - the forward modeled derivative with temperature
    """
    
    Nn,dNndB,dNndnWV,dNndT = O2DIALDerivative(T,P,nWV,BSR,rb_on,abs_on,Trx_on,inu0_on,GainRatio,dr)
    Nf,dNfdB,dNfdnWV,dNfdT = O2DIALDerivative(T,P,nWV,BSR,rb_off,abs_off,Trx_off,inu0_off,1.0,dr)
        
    ChannelRatio = Nn/Nf
    dCRdB = (Nf[np.newaxis]*dNndB-Nn[np.newaxis]*dNfdB)/(Nf[np.newaxis])**2
    dCRdT = (Nf[np.newaxis]*dNndT-Nn[np.newaxis]*dNfdT)/(Nf[np.newaxis])**2
    dCRdnWV = (Nf[np.newaxis]*dNndnWV-Nn[np.newaxis]*dNfdnWV)/(Nf[np.newaxis])**2
    
    
    return ChannelRatio,dCRdB,dCRdnWV,dCRdT
    
def HSRLProfileRatioDeriv(T,P,BSR,TrxM,TrxC,rb_spec,inu0,GainRatio=1.0):
    """
    Ratio is Combined/Molecular channels
    GainRatio: Combined Gain/Molecular Gain
    Inputs:
        T - Temperature in Kelvin
        P - Pressure in atm
        BSR - aerosol backscatter ratio
        TrxM - Receiver spectrum in the molecular channel
        TrxC - Receiver spectrum in the combined channel
        rb_spec - dictionary of Rayleigh-Brillioun PCA terms
        inu0 - index to center/transmit frequency on frequency grid
        Gain Ratio - Combined/Molecular relative gain
        
    Returns:
        ChannelRatio - the forward modeled ratio of the two channels
        dCRdB - the forward modeled derivative with backscatter ratio
        dCRdT - the forward modeled derivative with temperature
    """
    
    Nm,dNmdB,dNmdT = HSRLDerivative(T,P,BSR,rb_spec,TrxM,inu0,1.0)
    Nc,dNcdB,dNcdT = HSRLDerivative(T,P,BSR,rb_spec,TrxC,inu0,GainRatio)
    
    ChannelRatio = Nc/Nm
    dCRdB = (Nm[np.newaxis]*dNcdB-Nc[np.newaxis]*dNmdB)/(Nm[np.newaxis])**2
    dCRdT = (Nm[np.newaxis]*dNcdT-Nc[np.newaxis]*dNmdT)/(Nm[np.newaxis])**2
    
    
    return ChannelRatio,dCRdB,dCRdT
    
    
def HSRLDerivative(T,P,BSR,rb_spec,Trx,inu0,K):
    # RB sensitivity needed for temperature derivative    
    betaM,dbetaMdT = rb.calc_pca_spectrum_w_deriv(rb_spec,T,P)
#    betaMnorm = np.sum(betaM,axis=0)  # temporary fix to normalize molecular spectrum
#    betaM = betaM/betaMnorm
#    dbetaMdT = dbetaMdT/betaMnorm
    # HSRL Profile
    N = K*(Trx[inu0]*(1-1/BSR)+1/BSR*np.sum(Trx[:,np.newaxis]*betaM,axis=0))
    
    # derivative vs backscatter ratio
    dNdB = np.diag(K/BSR**2*(Trx[inu0]-np.sum(Trx[:,np.newaxis]*betaM,axis=0)))

    # derivative vs temperature
    dNdT = np.diag(K/BSR*np.sum(Trx[:,np.newaxis]*dbetaMdT,axis=0))
    
    return N,dNdB,dNdT

def WVDIALDerivative(T,P,nWV,BSR,rb_spec,abs_spec,Trx,inu0,K,dr):
    # RB sensitivity needed for temperature derivative    
    betaM,dbetaMdT = rb.calc_pca_spectrum_w_deriv(rb_spec,T,P)
#    betaMnorm = np.sum(betaM,axis=0)  # temporary fix to normalize molecular spectrum
#    betaM = betaM/betaMnorm
#    dbetaMdT = dbetaMdT/betaMnorm
    # Absorption sensitivity
    sigS,dsigSdT = rb.calc_pca_spectrum_w_deriv(abs_spec,T,P)
    ODs = np.cumsum(nWV*sigS,axis=1)*dr
    Ts = np.exp(-ODs)
    
    # These definitions are specific to water vapor
    # full 3D OD derivatives
    dODsdnWV = np.cumsum(sigS[:,:,np.newaxis]*np.eye(T.size)[np.newaxis,:,:],axis=1)*dr
    dODsdT = np.cumsum((nWV*dsigSdT)[:,:,np.newaxis]*np.eye(T.size)[np.newaxis,:,:],axis=1)*dr
    
    # laser line OD derivatives
    dODsdnWV_nuL = -np.cumsum(np.diag(sigS[inu0,:]),axis=1)*dr  
    dODsdT_nuL = -np.cumsum(np.diag(nWV*dsigSdT[inu0,:]),axis=1)*dr  
    
    # Calculate WV profile
    N = K*Ts[inu0,:]*(Ts[inu0,:]*Trx[inu0]*(1-1/BSR) \
        +1/BSR*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0))
        
    # derivative with respect to water vapor
    # full 3D matrix integral
    dNdnWV = N[np.newaxis,:]*dODsdnWV_nuL \
        +(K*Ts[inu0,:])[np.newaxis,:]*((Ts[inu0,:])[np.newaxis,:]*dODsdnWV_nuL*Trx[inu0]*(1-1/BSR[np.newaxis,:]) \
        -1/BSR[np.newaxis,:]*np.sum(Trx[:,np.newaxis,np.newaxis]*Ts[:,:,np.newaxis]*dODsdnWV*betaM[:,:,np.newaxis],axis=0).T)
        
    # derivative with respect to temperature
    # full 3D matrix integral
    dNdT = N[np.newaxis,:]*dODsdT_nuL \
        +(K*Ts[inu0,:])[np.newaxis,:]*((Ts[inu0,:])[np.newaxis,:]*dODsdT_nuL*Trx[inu0]*(1-1/BSR[np.newaxis,:]) \
        +1/BSR[np.newaxis,:]*(np.diag(np.sum(Trx[:,np.newaxis]*Ts*dbetaMdT,axis=0)) \
        -np.sum(Trx[:,np.newaxis,np.newaxis]*Ts[:,:,np.newaxis]*dODsdT*betaM[:,:,np.newaxis],axis=0).T))
    
    # derivative with respect to temperature
    dNdB = np.diag(K*Ts[inu0,:]*(Ts[inu0,:]*Trx[inu0]/BSR**2 \
        -1/BSR**2*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0)))
#        
    # derivative definitions:
    #   range axis - columns
    #   state parameter axis - rows
    return N,dNdB,dNdnWV,dNdT
    

def O2DIALDerivative(T,P,nWV,BSR,rb_spec,abs_spec,Trx,inu0,K,dr):
    # RB sensitivity needed for temperature derivative    
    betaM,dbetaMdT = rb.calc_pca_spectrum_w_deriv(rb_spec,T,P)
#    betaMnorm = np.sum(betaM,axis=0)  # temporary fix to normalize molecular spectrum
#    betaM = betaM/betaMnorm
#    dbetaMdT = dbetaMdT/betaMnorm
    # Absorption sensitivity
    sigS,dsigSdT = rb.calc_pca_spectrum_w_deriv(abs_spec,T,P)
    ODs = np.cumsum(fo2*sigS*(P/(lp.kB*T)-nWV)[np.newaxis,:],axis=1)*dr
    Ts = np.exp(-ODs)
    
    # These definitions are specific to water vapor
#    dODsdnWV = np.cumsum(fo2*sigS,axis=1)*dr
#    dODsdT = np.cumsum(fo2*P[np.newaxis,:]/(lp.kB*T[np.newaxis,:])*(dsigSdT-sigS/T[np.newaxis,:]),axis=1)
    
    # These definitions are specific to oxygen
    # full 3D OD derivatives
    dODsdnWV = -np.cumsum(fo2*sigS[:,:,np.newaxis]*np.eye(T.size)[np.newaxis,:,:],axis=1)*dr
#    dODsdT = np.cumsum((fo2*P[np.newaxis,:]/(lp.kB*T[np.newaxis,:])*(dsigSdT-sigS/T[np.newaxis,:])) \
#        [:,:,np.newaxis]*np.eye(T.size)[np.newaxis,:,:],axis=1)*dr
    dODsdT = np.cumsum((fo2*dsigSdT*(P/(lp.kB*T)-nWV)[np.newaxis,:]-fo2*sigS*(P/(lp.kB*T**2))[np.newaxis,:]) \
        [:,:,np.newaxis]*np.eye(T.size)[np.newaxis,:,:],axis=1)*dr
    
    # laser line OD derivatives
    dODsdnWV_nuL = np.cumsum(np.diag(fo2*sigS[inu0,:]),axis=1)*dr  

    dODsdT_nuL = -np.cumsum(np.diag(fo2*dsigSdT[inu0,:]*(P/(lp.kB*T)-nWV)-fo2*sigS[inu0,:]*P/(lp.kB*T**2)),axis=1)*dr
    
    # Calculate WV profile
    N = K*Ts[inu0,:]*(Ts[inu0,:]*Trx[inu0]*(1-1/BSR) \
        +1/BSR*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0))
      
      
    # derivative with respect to water vapor
    # full 3D matrix integral
    dNdnWV = N[np.newaxis,:]*dODsdnWV_nuL \
        +(K*Ts[inu0,:])[np.newaxis,:]*((Ts[inu0,:])[np.newaxis,:]*dODsdnWV_nuL*Trx[inu0]*(1-1/BSR[np.newaxis,:]) \
        -1/BSR[np.newaxis,:]*np.sum(Trx[:,np.newaxis,np.newaxis]*Ts[:,:,np.newaxis]*dODsdnWV*betaM[:,:,np.newaxis],axis=0).T)
        
    # derivative with respect to temperature
    # full 3D matrix integral
    dNdT = N[np.newaxis,:]*dODsdT_nuL \
        +(K*Ts[inu0,:])[np.newaxis,:]*((Ts[inu0,:])[np.newaxis,:]*dODsdT_nuL*Trx[inu0]*(1-1/BSR[np.newaxis,:]) \
        +1/BSR[np.newaxis,:]*(np.diag(np.sum(Trx[:,np.newaxis]*Ts*dbetaMdT,axis=0)) \
        -np.sum(Trx[:,np.newaxis,np.newaxis]*Ts[:,:,np.newaxis]*dODsdT*betaM[:,:,np.newaxis],axis=0).T))
    
    # derivative with respect to temperature
    dNdB = np.diag(K*Ts[inu0,:]*(Ts[inu0,:]*Trx[inu0]/BSR**2 \
        -1/BSR**2*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0)))
        
#    # derivative with respect to water vapor
#    dNdnWV = K*Ts[inu0,:]*(-dODsdnWV*(Ts[inu0,:]*Trx[inu0]*(1-1/BSR) \
#        +1/BSR*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0)) \
#        -(Ts[inu0,:]*dODsdnWV[inu0,:]*Trx[inu0]*(1-1/BSR) \
#        +1/BSR*np.sum(Trx[:,np.newaxis]*Ts*dODsdnWV*betaM,axis=0)))
#        
#    # derivative with respect to temperature
#    dNdT = K*Ts[inu0,:]*(-dODsdT[inu0,:]*(Trx[inu0]*(1-1/BSR) \
#        +1/BSR*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0)) \
#        -(Ts[inu0,:]*dODsdT[inu0,:]*Trx[inu0]*(1-1/BSR) \
#        +1/BSR*np.sum(Trx[:,np.newaxis]*Ts*(dODsdT*betaM+dbetaMdT),axis=0)))
#        
#    dNdB = K*Ts[inu0,:]*(Ts[inu0,:]*Trx[inu0]/BSR**2 \
#        -1/BSR**2*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0))
        
    return N,dNdB,dNdnWV,dNdT
  




def WaterVaporProfileRatio(T,P,nWV,BSR,Trx_on,Trx_off,rb_on,rb_off,abs_on,abs_off,dr,inu0_on,inu0_off,GainRatio=1.0):
    """
    Ratio is Online/Offline channels
    GainRatio: Online/Offline Gain
    Inputs:
        T - Temperature in Kelvin
        P - Pressure in atm
        nWV - water vapor density
        BSR - aerosol backscatter ratio
        Trx_on - Receiver spectrum in the online channel
        Trx_off - Receiver spectrum in the offline channel
        rb_on - dictionary of Rayleigh-Brillioun PCA terms for online wavelength
        rb_off - dictionary of Rayleigh-Brillioun PCA terms for offline wavlength
        abs_on - dictionary of Water Vapor spectrum PCA terms for online wavelength
        abs_off - dictionary of Water Vapor PCA terms for offline wavlength
        dr - range grid spacing in meters
        inu0 - index to center/transmit frequency on frequency grid
        Gain Ratio - Combined/Molecular relative gain
        
    Returns:
        ChannelRatio - the forward modeled ratio of the two channels
        dCRdB - the forward modeled derivative with backscatter ratio
        dCRdnWV - the forward modeled derivative with water vapor
        dCRdT - the forward modeled derivative with temperature
    """
    
    Nn = WVDIALProfile(T,P,nWV,BSR,rb_on,abs_on,Trx_on,inu0_on,GainRatio,dr)
    Nf = WVDIALProfile(T,P,nWV,BSR,rb_off,abs_off,Trx_off,inu0_off,1.0,dr)
        
    ChannelRatio = Nn/Nf

    
    return ChannelRatio
    
def OxygenProfileRatio(T,P,nWV,BSR,Trx_on,Trx_off,rb_on,rb_off,abs_on,abs_off,dr,inu0_on,inu0_off,GainRatio=1.0):
    """
    Ratio is Online/Offline channels
    GainRatio: Online/Offline Gain
    Inputs:
        T - Temperature in Kelvin
        P - Pressure in atm
        nWV - water vapor density
        BSR - aerosol backscatter ratio
        Trx_on - Receiver spectrum in the online channel
        Trx_off - Receiver spectrum in the offline channel
        rb_on - dictionary of Rayleigh-Brillioun PCA terms for online wavelength
        rb_off - dictionary of Rayleigh-Brillioun PCA terms for offline wavlength
        abs_on - dictionary of Oxygen spectrum PCA terms for online wavelength
        abs_off - dictionary of Oxygen PCA terms for offline wavlength
        dr - range grid spacing in meters
        inu0 - index to center/transmit frequency on frequency grid
        Gain Ratio - On/Off relative gain
        
    Returns:
        ChannelRatio - the forward modeled ratio of the two channels
        dCRdB - the forward modeled derivative with backscatter ratio
        dCRdnWV - the forward modeled derivative with water vapor
        dCRdT - the forward modeled derivative with temperature
    """
    
    Nn = O2DIALProfile(T,P,nWV,BSR,rb_on,abs_on,Trx_on,inu0_on,GainRatio,dr) # online
    Nf = O2DIALProfile(T,P,nWV,BSR,rb_off,abs_off,Trx_off,inu0_off,1.0,dr) # offline
        
    ChannelRatio = Nn/Nf
 
    return ChannelRatio
    
def HSRLProfileRatio(T,P,BSR,TrxM,TrxC,rb_spec,inu0,GainRatio=1.0):
    """
    Ratio is Combined/Molecular channels
    GainRatio: Combined Gain/Molecular Gain
    Inputs:
        T - Temperature in Kelvin
        P - Pressure in atm
        BSR - aerosol backscatter ratio
        TrxM - Receiver spectrum in the molecular channel
        TrxC - Receiver spectrum in the combined channel
        rb_spec - dictionary of Rayleigh-Brillioun PCA terms
        inu0 - index to center/transmit frequency on frequency grid
        Gain Ratio - Combined/Molecular relative gain
        
    Returns:
        ChannelRatio - the forward modeled ratio of the two channels
        dCRdB - the forward modeled derivative with backscatter ratio
        dCRdT - the forward modeled derivative with temperature
    """
    
    Nm = HSRLProfile(T,P,BSR,rb_spec,TrxM,inu0,1.0)  # molecular
    Nc = HSRLProfile(T,P,BSR,rb_spec,TrxC,inu0,GainRatio)  # combined
    
    ChannelRatio = Nc/Nm
  
    return ChannelRatio
    
    
def HSRLProfile(T,P,BSR,rb_spec,Trx,inu0,K):
    # RB sensitivity needed for temperature derivative    
    betaM = rb.calc_pca_spectrum(rb_spec,T,P)
#    betaM = betaM/np.sum(betaM,axis=0)  # temporary fix to normalize molecular spectrum    
    # HSRL Profile
    N = K*(Trx[inu0]*(1-1/BSR)+1/BSR*np.sum(Trx[:,np.newaxis]*betaM,axis=0))
    
    return N

def WVDIALProfile(T,P,nWV,BSR,rb_spec,abs_spec,Trx,inu0,K,dr):
    # RB sensitivity needed for temperature derivative    
    betaM = rb.calc_pca_spectrum(rb_spec,T,P)
#    betaM = betaM/np.sum(betaM,axis=0)  # temporary fix to normalize molecular spectrum
    # Absorption sensitivity
    sigS = rb.calc_pca_spectrum(abs_spec,T,P)
    ODs = np.cumsum(nWV*sigS,axis=1)*dr
    Ts = np.exp(-ODs)
    
    # Calculate WV profile
    N = K*Ts[inu0,:]*(Ts[inu0,:]*Trx[inu0]*(1-1/BSR) \
        +1/BSR*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0))
                
    return N
    

def O2DIALProfile(T,P,nWV,BSR,rb_spec,abs_spec,Trx,inu0,K,dr):
    # RB sensitivity needed for temperature derivative    
    betaM = rb.calc_pca_spectrum(rb_spec,T,P)
#    betaM = betaM/np.sum(betaM,axis=0)  # temporary fix to normalize molecular spectrum
    
    # Absorption sensitivity
    sigS = rb.calc_pca_spectrum(abs_spec,T,P)
    ODs = np.cumsum(fo2*sigS*(P/(lp.kB*T)-nWV)[np.newaxis,:],axis=1)*dr
    Ts = np.exp(-ODs)
    
    # Calculate WV profile
    N = K*Ts[inu0,:]*(Ts[inu0,:]*Trx[inu0]*(1-1/BSR) \
        +1/BSR*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0))
        
        
    return N

  
#def TD_ErrorFunction(T,P,B,n_wv,WV_Online,WV_OffLine,)

# Added to SpectrumLib Sept 15 2017
#def ExtinctionFromHITRAN(nu,TempProf,PresProf,Mmol,filename='',freqnorm=False,nuLim=np.array([])):
#    """
#    WV_ExtinctionFromHITRAN(nu,TempProf,PresProf)
#    returns a WV extinction profile in m^-1 for a given
#    nu - frequency grid in Hz
#    TempProf - Temperature array in K
#    PresProf - Pressure array in Atm (must be same size as TempProf)
#    
#    Note that the height of the extinction profile will change based on the
#    grid resolution of nu.  
#    Set freqnorm=True
#    To obtain a grid independent profile to obtain extinction in m^-1 Hz^-1
#    
#    This function requires access to the HITRAN ascii data:
#    '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/WV_HITRAN2012_815_841.txt'
#    The data file can be subtituted with something else by using the optional
#    filename input.  This accepts a string with a path to the desired file.
#    
#    If a full spectrum is not needed (nu only represents an on and off line),
#    use nuLim to define the frequency limits over which the spectral lines 
#    should be included.
#    nuLim should be a two element numpy.array
#    nuLim[0] = minimum frequency
#    nuLim[1] = maximum frequency
#    
#    """
#    nuL = np.mean(nu);
#    
#    if not filename:
#        filename = os.path.abspath(__file__+'/../DataFiles/') + '/WV_HITRAN2012_815_841.txt'
##        filename = '/h/eol/mhayman/PythonScripts/NCAR-LidarProcessing/libraries/WV_HITRAN2012_815_841.txt';
#    
##    Mh2o = (mH2O*1e-3)/N_A; # mass of a single water molecule, kg/mol
#    
#    # read HITRAN data
#    data = np.loadtxt(filename,delimiter=',',usecols=(0,1,2,3,4,5,6,7,8,9),skiprows=13)
#    
#    if nuLim.size == 0:
#        nuSpan = np.max(nu) - np.min(nu)
#        nuLim = np.array([np.min(nu)-nuSpan*0.0,np.max(nu)+nuSpan*0.0])
#    
#    #Voigt profile calculation
#    wn_nu  = nu/lp.c*1e-2; # convert to wave number in cm^-1
#    wn_nuL  = nuL/lp.c*1e-2; # convert laser frequency to wave number in cm^-1
#    wn_nuLim = nuLim/lp.c*1e-2  # convert frequency span of included lines to wave number in cm^-1
#    #Find lines from WNmin to WNmax to calculate voigt profile
#    hitran_line_indices = np.nonzero(np.logical_and(data[:,2] > wn_nuLim[0],data[:,2] < wn_nuLim[1]))[0];
##    print('%d'%hitran_line_indices.size)
#    
#    hitran_T00 = 296;              # HITRAN reference temperature [K]
#    hitran_P00 = 1;                # HITRAN reference pressure [atm]
#    hitran_nu0_0 = data[hitran_line_indices,2];      # absorption line center wavenumber from HITRAN [cm^-1]
#    hitran_S0 = data[hitran_line_indices,3];         # initial linestrength from HITRAN [cm^-1/(mol*cm^-2)]   
#    hitran_gammal0 = data[hitran_line_indices,5];    # air-broadened halfwidth at T_ref and P_ref from HITRAN [cm^-1/atm]
##    hitran_gamma_s = data[hitran_line_indices,6];    # self-broadened halfwidth at T_ref and P_ref from HITRAN [cm^-1/atm]
#    hitran_E = data[hitran_line_indices,7];          # ground state transition energy from HITRAN [cm^-1]  
#    hitran_alpha = data[hitran_line_indices,8];      # linewidth temperature dependence factor from HITRAN
#    hitran_delta = data[hitran_line_indices,9];     # pressure shift from HiTRAN [cm^-1 atm^-1]
#    
#    
#    voigt_sigmav_f = np.zeros((np.size(TempProf),np.size(wn_nu)));
#    
#    dnu = np.mean(np.diff(nu))
##    dnu_sign = np.sign(dnu)
#    dwn_nu = np.mean(np.diff(wn_nu))
#    
#    # calculate the absorption cross section at each range
#    for ai in range(np.size(TempProf)): 
#        #    %calculate the pressure shifts for selected lines as function of range
#        hitran_nu0 = hitran_nu0_0+hitran_delta*(PresProf[ai]/hitran_P00); # unclear if it should be Pi/P00
#        hitran_gammal = hitran_gammal0*(PresProf[ai]/hitran_P00)*((hitran_T00/TempProf[ai])**hitran_alpha);    # Calculate Lorentz linewidth at P(i) and T(i)
#        hitran_gammad = (hitran_nu0)*((2.0*lp.kB*TempProf[ai]*np.log(2.0))/(Mmol*lp.c**2))**(0.5);  # Calculate HWHM Doppler linewidth at T(i)                                        ^
#        
#        # term 1 in the Voigt profile
##        voigt_y = (hitran_gammal/hitran_gammad)*((np.log(2.0))**(0.5));
#        voigt_x_on = ((wn_nuL-hitran_nu0)/hitran_gammad)*(np.log(2.0))**(0.5);
#    
#        # setting up Voigt convolution
##        voigt_t = np.arange(-np.shape(hitran_line_indices)[0]/2.0,np.shape(hitran_line_indices)[0]/2); # set up the integration spectral step size
#        
#        voigt_f_t = np.zeros((np.size(hitran_line_indices),np.size(wn_nu)));
#        for bi in range(voigt_x_on.size):
#            voigt_f_t[bi,:] = voigt(wn_nu-hitran_nu0[bi],hitran_gammad[bi],hitran_gammal[bi],norm=True); 
#            if freqnorm:
#                voigt_f_t[bi,:] = voigt_f_t[bi,:]
##                voigt_f_t[bi,:] = dnu_sign*voigt_f_t[bi,:]/np.trapz(voigt_f_t[bi,:],x=nu);
#            else:
#                voigt_f_t[bi,:] = voigt_f_t[bi,:]*np.abs(dwn_nu)
##                voigt_f_t[bi,:] = voigt_f_t[bi,:]/np.trapz(voigt_f_t[bi,:]);  # add x=wn_nu to add frequency normalization
#    
#        # Calculate linestrength at temperature T
#        hitran_S = hitran_S0*((hitran_T00/TempProf[ai])**(1.5))*np.exp(1.439*hitran_E*((1.0/hitran_T00)-(1.0/TempProf[ai])));
#      
#        # Cross section is normalized for spectral integration (no dnu multiplier required)
#        voigt_sigmav_f[ai,:] = np.nansum(hitran_S[:,np.newaxis]*voigt_f_t,axis=0);  
#    
#    
#    ExtinctionProf = (1e-4)*voigt_sigmav_f;  # convert to m^2
#    return ExtinctionProf
#    
#
#def voigt(x,alpha,gamma,norm=True):
#    """
#    voigt(x,alpha,gamma)
#    Calculates a zero mean voigt profile for spectrum x 
#    alpha - Gaussian HWHM
#    gamma - lorentzian HWMM
#    norm - True: normalize area under the profile's curve
#           False:  max value of profile = 1
#    
#    for instances where the profile is not zero mean substitute x-xmean for x
#    
#    see scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
#    """
#    sigma = alpha / np.sqrt(2*np.log(2))
#    if norm:
#        v_prof = np.real(wofz((x+1j*gamma)/sigma/np.sqrt(2)))/sigma/np.sqrt(2*np.pi)
#        return v_prof
#    else:
#        v_prof = np.real(wofz((x+1j*gamma)/sigma/np.sqrt(2))) #np.pi/gamma
##        v_prof/ np.real(wofz((0.0+1j*gamma)/sigma/np.sqrt(2))) # normalize so V(x=0) = 1
##        v_prof = (np.pi*sigma/gamma*np.exp(gamma**2/sigma**2)*(1-scipy.special.erf(gamma/sigma)))*v_prof/ np.real(wofz((0.0+1j*gamma)/sigma/np.sqrt(2))) # normalize so V(x=0) = np.pi*sigma/gamma
##        v_prof = (np.pi*sigma/gamma)*v_prof/ np.real(wofz((0.0+1j*gamma)/sigma/np.sqrt(2))) # normalize so V(x=0) = np.pi*sigma/gamma
#        return v_prof
#        
#def load_spect_params(pca_file,nu = np.array([])):
#    data = np.load(pca_file)
#    M = data['M'].copy()
#    Mavg = data['Mavg'].copy()
#    nu_pca = data['sim_nu'].copy()
#    pT = data['pT'].copy()
#    pP = data['pP'].copy()
#    Tmean = data['Tmean']
#    Tstd = data['Tstd']
#    Pmean = data['Pmean']
#    Pstd = data['Pstd']
#    
#    pT = pT[np.newaxis,:]
#    pP = pP[np.newaxis,:]
##    pT0 = np.ones(pT.shape)
##    pT0[np.nonzero(pT==0)] = 0  # mask for terms that become zero in derivative
#    
#    if nu.size != 0:
#        Mint = np.zeros((nu.size,M.shape[1]))
#        if np.max(nu) > np.max(nu_pca) or np.min(nu) < np.min(nu_pca):
#            print('WARNING:  frequency limits in load_spect_params() exceeded for file %s'%pca_file)
#
#        for ai in range(M.shape[1]):
#            Mint[:,ai] = np.interp(nu,nu_pca,M[:,ai],left=0,right=0)
#        Mavg_int = np.matrix(np.interp(nu,nu_pca,np.array(Mavg).flatten(),left=0,right=0)[:,np.newaxis])
#        datalist = [Mint,Mavg_int,nu,pT,pP,Tmean,Tstd,Pmean,Pstd]
#    else:
#        datalist = [M,Mavg,nu_pca,pT,pP,Tmean,Tstd,Pmean,Pstd]
#    
#    labels = ['M','Mavg','nu_pca','pT','pP','Tmean','Tstd','Pmean','Pstd']
#    
#    spec_data = dict(zip(labels,datalist))
#    return spec_data
#    
#def calc_pca_spectrum(spec_data,T,P,nu=np.array([])):
#    """
#    Calculates extinction spectrum for T and P arrays (of equal size)
#    spec_data - dict of PCA spectral data returned by load_spect_params
#    if nu is unassigned, the data is returned at native resolution
#        (this is the fasted option)
#    if nu is defined, the data is interpolated to the requested grid
#    
#    If speed is needed it is not recomended that nu be provided.  Instead
#    provide the required frequency grid to load_spect_params.  That will
#    configure the PCA matrices so no interpolation is required.
#    """
#    TPmat = np.matrix(((T.flatten()-spec_data['Tmean'])/spec_data['Tstd'])[:,np.newaxis]**spec_data['pT'] \
#        *((P.flatten()-spec_data['Pmean'])/spec_data['Pstd'])[:,np.newaxis]**spec_data['pP'])
#    extinction = np.array(spec_data['M']*TPmat.T+spec_data['Mavg'])
#
#    if nu.size > 0:
#        # if nu is provided, interpolate to obtain requrested frequency grid
#        ext_interp = np.zeros((nu.size,T.size))
#   
#        for ai in range(T.size):
#            ext_interp[:,ai] = np.interp(nu,spec_data['nu_pca'],extinction[:,ai],left=0,right=0)
#        return ext_interp   
#    else:
#        return extinction
#        
#def calc_pca_spectrum_w_deriv(spec_data,T,P,nu=np.array([])):
#    """
#    Calculates extinction spectrum and its temperature derivative for T and P 
#    arrays (of equal size)
#    spec_data - dict of PCA spectral data returned by load_spect_params
#    if nu is unassigned, the data is returned at native resolution
#        (this is the fasted option)
#    if nu is defined, the data is interpolated to the requested grid
#    
#    If speed is needed it is not recomended that nu be provided.  Instead
#    provide the required frequency grid to load_spect_params.  That will
#    configure the PCA matrices so no interpolation is required.
#    """
#    TPmat = np.matrix(((T.flatten()-spec_data['Tmean'])/spec_data['Tstd'])[:,np.newaxis]**spec_data['pT'] \
#        *((P.flatten()-spec_data['Pmean'])/spec_data['Pstd'])[:,np.newaxis]**spec_data['pP'])
#    dTPmat = np.matrix(spec_data['pT']*(((T.flatten()-spec_data['Tmean'])/spec_data['Tstd'])[:,np.newaxis]**(spec_data['pT']-1) \
#        *((P.flatten()-spec_data['Pmean'])/spec_data['Pstd'])[:,np.newaxis]**spec_data['pP']))/spec_data['Tstd']
#
#    extinction = np.array(spec_data['M']*TPmat.T+spec_data['Mavg'])
#    dextinction = np.array(spec_data['M']*dTPmat.T)
#
#    if nu.size > 0:
#        # if nu is provided, interpolate to obtain requrested frequency grid
#        ext_interp = np.zeros((nu.size,T.size))
#        dext_interp = np.zeros((nu.size,T.size))
#   
#        for ai in range(T.size):
#            ext_interp[:,ai] = np.interp(nu,spec_data['nu_pca'],extinction[:,ai],left=0,right=0)
#            dext_interp[:,ai] = np.interp(nu,spec_data['nu_pca'],dextinction[:,ai],left=0,right=0)
#        return ext_interp,dext_interp
#    else:
#        return extinction,dextinction
#
#def load_RB_params(filename=''):
#    """
#    RB_Spectrum(T,P,lam,nu=np.array([]))
#    Obtain the Rayleigh-Brillouin Spectrum of Earth atmosphere
#    T - Temperature in K.  Accepts an array
#    P - Pressure in Atm.  Accepts an array with size equal to size of T
#    lam - wavelength in m
#    nu - differential frequency basis.  If not supplied uses native frequency from the PCA
#        analysis.
#    
#    
#    """
#
#    #Load results from PCA analysis (RB_PCA.m)
#    # Loads M, Mavg, x1d os.path.abspath(__file__+'/../../calibrations/')
#    if len(filename) == 0:
#        # default file search location
#        filename = os.path.abspath(__file__+'/../DataFiles/') + '/RB_PCA_Params.npz'
#    RBpca = np.load(filename);
#    M = RBpca['M']
#    Mavg = RBpca['Mavg']
#    x = RBpca['x']
#    dM = RBpca['dM']
#    RBpca.close()
#    
#    datalist = [M,Mavg,dM,x]
#    labels = ['M','Mavg','dM','x']
#    
#    RB_data = dict(zip(labels,datalist))
#    
#    return RB_data
#
#def calc_pca_RB_w_deriv(RB_data,lam,T,P,nu=np.array([]),norm=True):
#    # Obtain the y parameters from inputs
#    yR = lp.RayleighBrillouin_Y(T,P,lam);
#    
#    # Calculate spectrum based from yR and PCA data
#    yR = yR.flatten()
#    pow_y = np.arange(RB_data['M'].shape[1])[:,np.newaxis]
#    yvec = yR[np.newaxis,:]**pow_y
#    Spect = RB_data['Mavg']+np.dot(RB_data['M'],yvec)
#    
#    dxdT = -0.5*RB_data['x']/T[np.newaxis,:]
#    dydT = -0.5*(yR/T)[np.newaxis,:] # use 1st order term to obtain 
#    dyvecdT = dydT*pow_y*yR[np.newaxis,:]**(pow_y-1)
#    
##    Myprod = np.dot(RB_data['dM'],yvec)
##    dSpect = Myprod*dxdT+Myprod*dydT+np.dot(RB_data['M'],dyvecdT)
#    
##    dSpect = np.dot(RB_data['dM'],yvec)*dxdT+np.dot(RB_data['dM']*dydT,yvec)+np.dot(RB_data['M'],dyvecdT)
#    dSpect = np.dot(RB_data['dM'],yvec)*dxdT+np.dot(RB_data['M'],dyvecdT)
#
#    
#
#    if nu.size > 0:
#        # if nu is provided, interpolate to obtain requrested frequency grid
#        xR = lp.RayleighBrillouin_X(T,lam,nu)
#        SpcaI = np.zeros(xR.shape)
#        dSpcaI = np.zeros(xR.shape)
#   
#        for ai in range(T.size):
#            SpcaI[:,ai] = np.interp(xR[:,ai],RB_data['x'].flatten(),Spect[:,ai],left=0,right=0)
#            dSpcaI[:,ai] = np.interp(xR[:,ai],RB_data['x'].flatten(),dSpect[:,ai],left=0,right=0)
#        if norm:
#            normC = np.sum(SpcaI,axis=0)
#            SpcaI = SpcaI/normC[np.newaxis,:]
#            dSpcaI = dSpcaI/normC[np.newaxis,:]
#        return SpcaI,dSpcaI
#        
#    else:
#        # if nu is not provided, return the spectra and the native x axis
#        if norm:
#            normC = np.sum(Spect,axis=0)[np.newaxis,:]
#            Spect = Spect/normC
#            dSpect = dSpect/normC
#            
#        return Spect,dSpect,RB_data['x']
#
#def calc_pca_RB(RB_data,lam,T,P,nu=np.array([]),norm=True):
#    # Obtain the y parameters from inputs
#    yR = lp.RayleighBrillouin_Y(T,P,lam);
#    
#    # Calculate spectrum based from yR and PCA data
#    yR = yR.flatten()
#    pow_y = np.arange(RB_data['M'].shape[1])[:,np.newaxis]
#    yvec = yR[np.newaxis,:]**pow_y
#    Spect = RB_data['Mavg']+np.dot(RB_data['M'],yvec)
#
#    
#
#    if nu.size > 0:
#        # if nu is provided, interpolate to obtain requrested frequency grid
#        xR = lp.RayleighBrillouin_X(T,lam,nu)
#        SpcaI = np.zeros(xR.shape)
#   
#        for ai in range(T.size):
#            SpcaI[:,ai] = np.interp(xR[:,ai],RB_data['x'].flatten(),Spect[:,ai],left=0,right=0)
#        if norm:
#            normC = np.sum(SpcaI,axis=0)
#            SpcaI = SpcaI/normC[np.newaxis,:]
#        return SpcaI
#        
#    else:
#        # if nu is not provided, return the spectra and the native x axis
#        if norm:
#            normC = np.sum(Spect,axis=0)[np.newaxis,:]
#            Spect = Spect/normC
#            
#        return Spect,RB_data['x']