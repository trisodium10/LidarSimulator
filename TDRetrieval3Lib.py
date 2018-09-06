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

def TDProfiles(Prof,x,Trx,rb_spec,abs_spec,dr,inu0,bsrMult,base_T,base_P,r0):
    """
    Prof - dict of profiles to be reconstructed
    x - optimzation variable containing ln(T,nWV,BSR) as well as gain coefficients
    Trx - dict containing all the receiver transmission functions
    rb_spec - dict containing all the RB PCA data for all the channels
    abs_spec - dict containing all the absoprtion PCA data for DIAL channels
    dr - range bin size
    inu0 - dict containing the index to the center frequency of each channel
    bsrMult - dict containing the BSR multiplier due to wavelength difference
    base_T - surface temperature at lidar
    base_P - surface pressure at lidar
    r0 - range to first retrieval gate
    
    dict entries:
        'HSRL Mol'
        'HSRL Comb'
        'WV Online'
        'WV Offline'
        'O2 Online'
        'O2 Offline'
    profile entries:
        'HSRL Mol'
        'HSRL Comb'
        'WV Online'
        'WV Offline'
        'O2 Online'
        'O2 Offline'
    """
    
    iR = Prof['WV Online'].size # range index for a profile into 1D x array
    x2 = np.reshape(x,(iR+1,6))
    xK = x2[0,:]    # constants [HSRL Mol HSRL Comb, WV On, WV Off, O2 On ,O2 Off]
    xS = x2[1:,:]  # state vector [T, nWV, BSR, phi_HSRL, phi_WV, phi_O2]
    
    # HSRLProfile(T,BSR,phi,rb_spec,Trx,inu0,K,base_T,base_P)
    HSRL_mol = HSRLProfile(xS[:,0],xS[:,2],xS[:,3],rb_spec['HSRL'],Trx['HSRL Mol'],inu0['HSRL'],xK[0],base_T,base_P)
    HSRL_comb = HSRLProfile(xS[:,0],xS[:,2],xS[:,3],rb_spec['HSRL'],Trx['HSRL Comb'],inu0['HSRL'],xK[1],base_T,base_P)
    
#    plt.figure()
#    plt.plot(np.exp(xS[:,0]))
#    plt.title('Temperature [K]')
#    
#    plt.figure()
#    plt.semilogy(np.exp(xS[:,1]))
#    plt.title('WV number density [$m^{-3}$]')
#    
#    plt.figure()
#    plt.semilogy(np.exp(xS[:,2])+1)
#    plt.title('Backscatter Ratio')
    
    
#    HSRLModel = HSRLProfileRatio(xS[:,0],P,xS[:,2], \
#        Trx['HSRL Mol'],Trx['HSRL Comb'], \
#        rb_spec['HSRL'],inu0['HSRL'],GainRatio=xK[0])

#    WVDIALProfile(T,nWV,BSR,phi,rb_spec,abs_spec,Trx,inu0,K,base_T,base_P,dr)
    WV_on = WVDIALProfile(xS[:,0],xS[:,1],xS[:,2]+bsrMult['WV'],xS[:,4],rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0['WV Online'],xK[2],base_T,base_P,dr,r0)
    WV_off = WVDIALProfile(xS[:,0],xS[:,1],xS[:,2]+bsrMult['WV'],xS[:,4],rb_spec['WV Offline'],abs_spec['WV Offline'],Trx['WV Offline'],inu0['WV Offline'],xK[3],base_T,base_P,dr,r0)

#    WVModel = WaterVaporProfileRatio(xS[:,0],P,xS[:,1],xS[:,2]*bsrMult['WV'],
#        Trx['WV Online'], Trx['WV Offline'], \
#        rb_spec['WV Online'],rb_spec['WV Offline'], \
#        abs_spec['WV Online'],abs_spec['WV Offline'],dr, \
#        inu0['WV Online'],inu0['WV Offline'],GainRatio=xK[1])


#    O2DIALProfile(T,nWV,BSR,phi,rb_spec,abs_spec,Trx,inu0,K,base_T,base_P,dr)
    O2_on = O2DIALProfile(xS[:,0],xS[:,1],xS[:,2]+bsrMult['O2'],xS[:,5],rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0['O2 Online'],xK[4],base_T,base_P,dr,r0)
    O2_off = O2DIALProfile(xS[:,0],xS[:,1],xS[:,2]+bsrMult['O2'],xS[:,5],rb_spec['O2 Offline'],abs_spec['O2 Offline'],Trx['O2 Offline'],inu0['O2 Offline'],xK[5],base_T,base_P,dr,r0)
        
#    O2Model = OxygenProfileRatio(xS[:,0],P,xS[:,1],xS[:,2]*bsrMult['O2'],
#        Trx['O2 Online'], Trx['O2 Offline'], \
#        rb_spec['O2 Online'],rb_spec['O2 Offline'], \
#        abs_spec['O2 Online'],abs_spec['O2 Offline'],dr, \
#        inu0['O2 Online'],inu0['O2 Offline'],GainRatio=xK[2])
        
    return HSRL_mol, HSRL_comb, WV_on, WV_off, O2_on, O2_off

def TDErrorFunction(Prof,x,Trx,rb_spec,abs_spec,dr,inu0,bsrMult,base_T,base_P,r0,lam=[0,0,0,0,0,0]):
    """
    Prof - dict of ratio profiles to be reconstructed
    x - optimzation variable containing ln(T,nWV,BSR) as well as gain coefficients
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
        'HSRL Mol'
        'HSRL Comb'
        'WV Online'
        'WV Offline'
        'O2 Online'
        'O2 Offline'
    """
    
    iR = Prof['WV Online'].size # range index for a profile into 1D x array
    x2 = np.reshape(x,(iR+1,6))
    xK = x2[0,:]    # constants [HSRL Mol HSRL Comb, WV On, WV Off, O2 On ,O2 Off]
    xS = x2[1:,:]  # state vector [T, nWV, BSR, phi_HSRL, phi_WV, phi_O2]
    
    # HSRLProfile(T,BSR,phi,rb_spec,Trx,inu0,K,base_T,base_P)
    HSRL_mol = HSRLProfile(xS[:,0],xS[:,2],xS[:,3],rb_spec['HSRL'],Trx['HSRL Mol'],inu0['HSRL'],xK[0],base_T,base_P)+Prof['HSRL Mol BG']
    HSRL_comb = HSRLProfile(xS[:,0],xS[:,2],xS[:,3],rb_spec['HSRL'],Trx['HSRL Comb'],inu0['HSRL'],xK[1],base_T,base_P)+Prof['HSRL Comb BG']
    
#    WVDIALProfile(T,nWV,BSR,phi,rb_spec,abs_spec,Trx,inu0,K,base_T,base_P,dr)
    WV_on = WVDIALProfile(xS[:,0],xS[:,1],xS[:,2]+bsrMult['WV'],xS[:,4],rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0['WV Online'],xK[2],base_T,base_P,dr,r0)+Prof['WV Online BG']
    WV_off = WVDIALProfile(xS[:,0],xS[:,1],xS[:,2]+bsrMult['WV'],xS[:,4],rb_spec['WV Offline'],abs_spec['WV Offline'],Trx['WV Offline'],inu0['WV Offline'],xK[3],base_T,base_P,dr,r0)+Prof['WV Offline BG']

#    O2DIALProfile(T,nWV,BSR,phi,rb_spec,abs_spec,Trx,inu0,K,base_T,base_P,dr)
    O2_on = O2DIALProfile(xS[:,0],xS[:,1],xS[:,2]+bsrMult['O2'],xS[:,5],rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0['O2 Online'],xK[4],base_T,base_P,dr,r0)+Prof['O2 Online BG']
    O2_off = O2DIALProfile(xS[:,0],xS[:,1],xS[:,2]+bsrMult['O2'],xS[:,5],rb_spec['O2 Offline'],abs_spec['O2 Offline'],Trx['O2 Offline'],inu0['O2 Offline'],xK[5],base_T,base_P,dr,r0)+Prof['O2 Offline BG']
        
#    # Optimization error.  T is piecewise
#    OptError = np.nansum(HSRL_mol-(Prof['HSRL Mol'])*np.log(HSRL_mol)) \
#        +np.nansum(HSRL_comb-(Prof['HSRL Comb'])*np.log(HSRL_comb)) \
#        +np.nansum(WV_on-(Prof['WV Online'])*np.log(WV_on)) \
#        +np.nansum(WV_off-(Prof['WV Offline'])*np.log(WV_off)) \
#        +np.nansum(O2_on-(Prof['O2 Online'])*np.log(O2_on)) \
#        +np.nansum(O2_off-(Prof['O2 Offline'])*np.log(O2_off)) \
#        +lam[0]*np.nansum(np.abs(np.diff(xS[:,0]))) \
#        +lam[1]*np.nansum(np.abs(np.diff(xS[:,1]))) \
#        +lam[2]*np.nansum(np.abs(np.diff(xS[:,2]))) \
#        +lam[3]*np.nansum(np.abs(np.diff(xS[:,3]))) \
#        +lam[4]*np.nansum(np.abs(np.diff(xS[:,4]))) \
#        +lam[5]*np.nansum(np.abs(np.diff(xS[:,5]))) 
    
    # Optimization error.  T is piecewise slope
    OptError = np.nansum(HSRL_mol-(Prof['HSRL Mol'])*np.log(HSRL_mol)) \
        +np.nansum(HSRL_comb-(Prof['HSRL Comb'])*np.log(HSRL_comb)) \
        +np.nansum(WV_on-(Prof['WV Online'])*np.log(WV_on)) \
        +np.nansum(WV_off-(Prof['WV Offline'])*np.log(WV_off)) \
        +np.nansum(O2_on-(Prof['O2 Online'])*np.log(O2_on)) \
        +np.nansum(O2_off-(Prof['O2 Offline'])*np.log(O2_off)) \
        +lam[0]*np.nansum(np.abs(np.diff(np.diff(xS[:,0])))) \
        +lam[1]*np.nansum(np.abs(np.diff(xS[:,1]))) \
        +lam[2]*np.nansum(np.abs(np.diff(xS[:,2]))) \
        +lam[3]*np.nansum(np.abs(np.diff(xS[:,3]))) \
        +lam[4]*np.nansum(np.abs(np.diff(xS[:,4]))) \
        +lam[5]*np.nansum(np.abs(np.diff(xS[:,5])))
        
    return OptError

def TDGradientFunction(Prof,x,Trx,rb_spec,abs_spec,dr,inu0,bsrMult,base_T,base_P,r0,lam=[0,0,0,0,0,0]):
    """
    Prof - dict of ratio profiles to be reconstructed
    x - optimzation variable containing ln(T,nWV,BSR) as well as gain coefficients
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
        'HSRL Mol'
        'HSRL Comb'
        'WV Online'
        'WV Offline'
        'O2 Online'
        'O2 Offline'
    """ 
    
    iR = Prof['WV Online'].size # range index for a profile into 1D x array
    x2 = np.reshape(x,(iR+1,6))
    xK = x2[0,:]    # constants [HSRL Mol HSRL Comb, WV On, WV Off, O2 On ,O2 Off]
    xS = x2[1:,:]  # state vector [T, nWV, BSR, phi_HSRL, phi_WV, phi_O2]
    
    grad2 = np.zeros(x2.shape)   
    
    #N,dNdB,dNdT = HSRLDerivative(T,BSR,phi,rb_spec,Trx,inu0,K,base_T,base_P)
    HSRL_mol,dHmdB,dHmdT = HSRLDerivative(xS[:,0],xS[:,2],xS[:,3],rb_spec['HSRL'],Trx['HSRL Mol'],inu0['HSRL'],xK[0],base_T,base_P)
    HSRL_comb,dHcdB,dHcdT = HSRLDerivative(xS[:,0],xS[:,2],xS[:,3],rb_spec['HSRL'],Trx['HSRL Comb'],inu0['HSRL'],xK[1],base_T,base_P)
    
    # N,dNdB,dNdnWV,dNdT = WVDIALDerivative(T,nWV,BSR,phi,rb_spec,abs_spec,Trx,inu0,K,base_T,base_P,dr)
    WV_on,dWVndB,dWVndnWV,dWVndT = WVDIALDerivative(xS[:,0],xS[:,1],xS[:,2]+bsrMult['WV'],xS[:,4],rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0['WV Online'],xK[2],base_T,base_P,dr,r0)
    WV_off,dWVfdB,dWVfdnWV,dWVfdT = WVDIALDerivative(xS[:,0],xS[:,1],xS[:,2]+bsrMult['WV'],xS[:,4],rb_spec['WV Offline'],abs_spec['WV Offline'],Trx['WV Offline'],inu0['WV Offline'],xK[3],base_T,base_P,dr,r0)    
    
    # N,dNdB,dNdnWV,dNdT = O2DIALDerivative(T,nWV,BSR,phi,rb_spec,abs_spec,Trx,inu0,K,base_T,base_P,dr)
    O2_on,dO2ndB,dO2ndnWV,dO2ndT = O2DIALDerivative(xS[:,0],xS[:,1],xS[:,2]+bsrMult['O2'],xS[:,5],rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0['O2 Online'],xK[4],base_T,base_P,dr,r0)
    O2_off,dO2fdB,dO2fdnWV,dO2fdT = O2DIALDerivative(xS[:,0],xS[:,1],xS[:,2]+bsrMult['O2'],xS[:,5],rb_spec['O2 Offline'],abs_spec['O2 Offline'],Trx['O2 Offline'],inu0['O2 Offline'],xK[5],base_T,base_P,dr,r0)
    
#    HSRLModel,dHSdB,dHSdT = HSRLProfileRatioDeriv(xS[:,0],P,xS[:,2], \
#        Trx['HSRL Mol'],Trx['HSRL Comb'], \
#        rb_spec['HSRL'],inu0['HSRL'],GainRatio=xK[0])
#
#    WVModel,dWVdB,dWVdnWV,dWVdT = WaterVaporProfileRatioDeriv(xS[:,0],P,xS[:,1],xS[:,2]*bsrMult['WV'],
#        Trx['WV Online'], Trx['WV Offline'], \
#        rb_spec['WV Online'],rb_spec['WV Offline'], \
#        abs_spec['WV Online'],abs_spec['WV Offline'],dr, \
#        inu0['WV Online'],inu0['WV Offline'],GainRatio=xK[1])
#        
#    O2Model,dO2dB,dO2dnWV,dO2dT = OxygenProfileRatioDeriv(xS[:,0],P,xS[:,1],xS[:,2]*bsrMult['O2'],
#        Trx['O2 Online'], Trx['O2 Offline'], \
#        rb_spec['O2 Online'],rb_spec['O2 Offline'], \
#        abs_spec['O2 Online'],abs_spec['O2 Offline'],dr, \
#        inu0['O2 Online'],inu0['O2 Offline'],GainRatio=xK[2])
        
    HSRLmolBase = 1-(Prof['HSRL Mol'])/(HSRL_mol+Prof['HSRL Mol BG'])
    HSRLcombBase = 1-(Prof['HSRL Comb'])/(HSRL_comb+Prof['HSRL Comb BG'])
    WVonBase = 1-(Prof['WV Online'])/(WV_on+Prof['WV Online BG'])
    WVoffBase = 1-(Prof['WV Offline'])/(WV_off+Prof['WV Offline BG'])
    O2onBase = 1-(Prof['O2 Online'])/(O2_on+Prof['O2 Online BG'])
    O2offBase = 1-(Prof['O2 Offline'])/(O2_off+Prof['O2 Offline BG'])
    
    
#    HSRLbase = 2*(HSRLModel-Prof['HSRL'])/ProfVar['HSRL']
#    WVbase = 2*(WVModel-Prof['WV'])/ProfVar['WV']
#    O2base = 2*(O2Model-Prof['O2'])/ProfVar['O2']
    
    # temperature gradient
    grad2[1:,0] = np.nansum(HSRLmolBase[np.newaxis]*dHmdT,axis=1) \
        + np.nansum(HSRLcombBase[np.newaxis]*dHcdT,axis=1) \
        + np.nansum(WVonBase[np.newaxis]*dWVndT,axis=1) \
        + np.nansum(WVoffBase[np.newaxis]*dWVfdT,axis=1) \
        + np.nansum(O2onBase[np.newaxis]*dO2ndT,axis=1) \
        + np.nansum(O2offBase[np.newaxis]*dO2fdT,axis=1)
#    # piece wise penalty function    
#    gradpen = lam[0]*np.sign(np.diff(xS[:,0]))
#    gradpen[np.nonzero(np.isnan(gradpen))] = 0
#    grad2[2:,0] = grad2[2:,0] + gradpen
#    grad2[1:-1,0] = grad2[1:-1,0] - gradpen
#     piece wise slope penalty function    
    gradpen = lam[0]*np.sign(np.diff(np.diff(xS[:,0])))
    gradpen[np.nonzero(np.isnan(gradpen))] = 0
    grad2[3:,0] = grad2[3:,0] + gradpen
    grad2[2:-1,0] = grad2[2:-1,0] - 2*gradpen
    grad2[1:-2,0] = grad2[1:-2,0] + gradpen
    
    # water vapor gradient
    grad2[1:,1] = np.nansum(WVonBase[np.newaxis]*dWVndnWV,axis=1) \
        + np.nansum(WVoffBase[np.newaxis]*dWVfdnWV,axis=1) \
        + np.nansum(O2onBase[np.newaxis]*dO2ndnWV,axis=1) \
        + np.nansum(O2offBase[np.newaxis]*dO2fdnWV,axis=1)
    # piecewise penalty function
    gradpen = lam[1]*np.sign(np.diff(xS[:,1]))
    gradpen[np.nonzero(np.isnan(gradpen))] = 0
    grad2[2:,1] = grad2[2:,1] + gradpen
    grad2[1:-1,1] = grad2[1:-1,1] - gradpen
    
    # backscatter gradient
    grad2[1:,2] = np.nansum(HSRLmolBase[np.newaxis]*dHmdB,axis=1) \
        + np.nansum(HSRLcombBase[np.newaxis]*dHcdB,axis=1) \
        + np.nansum(WVonBase[np.newaxis]*dWVndB,axis=1) \
        + np.nansum(WVoffBase[np.newaxis]*dWVfdB,axis=1) \
        + np.nansum(O2onBase[np.newaxis]*dO2ndB,axis=1) \
        + np.nansum(O2offBase[np.newaxis]*dO2fdB,axis=1)   
#    #piecewise penalty function
#    gradpen = lam[2]*np.sign(np.diff(xS[:,2]))
#    gradpen[np.nonzero(np.isnan(gradpen))] = 0
#    grad2[2:,2] = grad2[2:,2] + gradpen
#    grad2[1:-1,2] = grad2[1:-1,2] - gradpen
    

    # *bsrMult['WV']
    # *bsrMult['WV']
    # *bsrMult['O2']
    # *bsrMult['O2']

    # HSRL Common terms
    grad2[1:,3] = np.nansum(HSRLmolBase[np.newaxis]*HSRL_mol,axis=0) + np.nansum(HSRLcombBase[np.newaxis]*HSRL_comb,axis=0)
#    # piece wise penalty function    
#    gradpen = lam[3]*np.sign(np.diff(xS[:,3]))
#    gradpen[np.nonzero(np.isnan(gradpen))] = 0
#    grad2[2:,3] = grad2[2:,3] + gradpen
#    grad2[1:-1,3] = grad2[1:-1,3] - gradpen
    
    # WV Common terms
    grad2[1:,4] = np.nansum(WVonBase[np.newaxis]*WV_on,axis=0) + np.nansum(WVoffBase[np.newaxis]*WV_off,axis=0)
#    # piece wise penalty function    
#    gradpen = lam[4]*np.sign(np.diff(xS[:,4]))
#    gradpen[np.nonzero(np.isnan(gradpen))] = 0
#    grad2[2:,4] = grad2[2:,4] + gradpen
#    grad2[1:-1,4] = grad2[1:-1,4] - gradpen
    
    # O2 Common terms
    grad2[1:,5] = np.nansum(O2onBase[np.newaxis]*O2_on,axis=0) + np.nansum(O2offBase[np.newaxis]*O2_off,axis=0)
#    # piece wise penalty function    
#    gradpen = lam[5]*np.sign(np.diff(xS[:,5]))
#    gradpen[np.nonzero(np.isnan(gradpen))] = 0
#    grad2[2:,5] = grad2[2:,5] + gradpen
#    grad2[1:-1,5] = grad2[1:-1,5] - gradpen

    grad2[0,0] = np.nansum(HSRLmolBase*HSRL_mol/xK[0])
    grad2[0,1] = np.nansum(HSRLcombBase*HSRL_comb/xK[1])
    grad2[0,2] = np.nansum(WVonBase*WV_on/xK[2])
    grad2[0,3] = np.nansum(WVoffBase*WV_off/xK[3])
    grad2[0,4] = np.nansum(O2onBase*O2_on/xK[4])
    grad2[0,5] = np.nansum(O2offBase*O2_off/xK[5])
    
#    grad2[0,1] = np.nansum(WVbase*WVModel/xK[1])
#    grad2[0,2] = np.nansum(O2base*O2Model/xK[2])
    
#    OptError = np.nansum(2*(HSRLModel-Prof['HSRL'])/ProfVar['HSRL']*) \
#        +np.nansum((WVModel-Prof['WV'])**2/ProfVar['WV']) \
#        +np.sum((O2Model-Prof['O2'])**2/ProfVar['O2'])
        
    return grad2.flatten()

    
    
def HSRLDerivative(lnT,lnBSR,phi,rb_spec,Trx,inu0,K,base_T,base_P):
    BSR = np.exp(lnBSR)+1
#    T = np.exp(lnT)
    T = lnT
    # RB sensitivity needed for temperature derivative    
    betaM,dbetaMdT = calc_pca_spectrum_w_deriv(rb_spec,T,base_T,base_P)
#    betaMnorm = np.sum(betaM,axis=0)  # temporary fix to normalize molecular spectrum
#    betaM = betaM/betaMnorm
#    dbetaMdT = dbetaMdT/betaMnorm
    # HSRL Profile
    N = K*np.exp(phi)*(Trx[inu0]*(1-1/BSR)+1/BSR*np.sum(Trx[:,np.newaxis]*betaM,axis=0))
    
    # derivative vs backscatter ratio
    dNdB = np.diag(K*np.exp(phi)/BSR**2*(Trx[inu0]-np.sum(Trx[:,np.newaxis]*betaM,axis=0)))*np.exp(lnBSR)[:,np.newaxis]

    # derivative vs temperature
#    dNdT = np.diag(K*np.exp(phi)/BSR*np.sum(Trx[:,np.newaxis]*dbetaMdT,axis=0))*T[:,np.newaxis]   # T exponent
    dNdT = np.diag(K*np.exp(phi)/BSR*np.sum(Trx[:,np.newaxis]*dbetaMdT,axis=0))                   # T
    
    # derivative vs phi is N
    
    return N,dNdB,dNdT

def WVDIALDerivative(lnT,lnnWV,lnBSR,phi,rb_spec,abs_spec,Trx,inu0,K,base_T,base_P,dr,r0):
    BSR = np.exp(lnBSR)+1
#    T = np.exp(lnT)
    T = lnT
    nWV = np.exp(lnnWV)       
    
    # RB sensitivity needed for temperature derivative    
    betaM,dbetaMdT = calc_pca_spectrum_w_deriv(rb_spec,T,base_T,base_P)
#    betaMnorm = np.sum(betaM,axis=0)  # temporary fix to normalize molecular spectrum
#    betaM = betaM/betaMnorm
#    dbetaMdT = dbetaMdT/betaMnorm
    # Absorption sensitivity
    sigS,dsigSdT = calc_pca_spectrum_w_deriv(abs_spec,T,base_T,base_P)
    dr_int = np.ones((1,nWV.size))*dr
    dr_int[0,0] = r0
    ODs = np.cumsum(nWV*sigS*dr_int,axis=1)
    Ts = np.exp(-ODs)
    
    # These definitions are specific to water vapor
    # full 3D OD derivatives
    dODsdnWV = np.cumsum((sigS*dr_int)[:,:,np.newaxis]*np.eye(T.size)[np.newaxis,:,:],axis=1)
    dODsdT = np.cumsum((nWV*dsigSdT*dr_int)[:,:,np.newaxis]*np.eye(T.size)[np.newaxis,:,:],axis=1)
    
    # laser line OD derivatives
    dODsdnWV_nuL = -np.cumsum(np.diag(sigS[inu0,:]*dr_int.flatten()),axis=1)
    dODsdT_nuL = -np.cumsum(np.diag(nWV*dsigSdT[inu0,:]*dr_int.flatten()),axis=1)
    
    # Calculate WV profile
    N = K*np.exp(phi)*Ts[inu0,:]*(Ts[inu0,:]*Trx[inu0]*(1-1/BSR) \
        +1/BSR*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0))
        
    # derivative with respect to water vapor exponent
    # full 3D matrix integral
    dNdnWV = (N[np.newaxis,:]*dODsdnWV_nuL \
        +(K*np.exp(phi)*Ts[inu0,:])[np.newaxis,:]*((Ts[inu0,:])[np.newaxis,:]*dODsdnWV_nuL*Trx[inu0]*(1-1/BSR[np.newaxis,:]) \
        -1/BSR[np.newaxis,:]*np.sum(Trx[:,np.newaxis,np.newaxis]*Ts[:,:,np.newaxis]*dODsdnWV*betaM[:,:,np.newaxis],axis=0).T))*nWV[:,np.newaxis]
        
    # derivative with respect to temperature exponent
    # full 3D matrix integral
#    dNdT = (N[np.newaxis,:]*dODsdT_nuL \
#        +(K*np.exp(phi)*Ts[inu0,:])[np.newaxis,:]*((Ts[inu0,:])[np.newaxis,:]*dODsdT_nuL*Trx[inu0]*(1-1/BSR[np.newaxis,:]) \
#        +1/BSR[np.newaxis,:]*(np.diag(np.sum(Trx[:,np.newaxis]*Ts*dbetaMdT,axis=0)) \
#        -np.sum(Trx[:,np.newaxis,np.newaxis]*Ts[:,:,np.newaxis]*dODsdT*betaM[:,:,np.newaxis],axis=0).T)))*T[:,np.newaxis] # T exponent
    dNdT = (N[np.newaxis,:]*dODsdT_nuL \
        +(K*np.exp(phi)*Ts[inu0,:])[np.newaxis,:]*((Ts[inu0,:])[np.newaxis,:]*dODsdT_nuL*Trx[inu0]*(1-1/BSR[np.newaxis,:]) \
        +1/BSR[np.newaxis,:]*(np.diag(np.sum(Trx[:,np.newaxis]*Ts*dbetaMdT,axis=0)) \
        -np.sum(Trx[:,np.newaxis,np.newaxis]*Ts[:,:,np.newaxis]*dODsdT*betaM[:,:,np.newaxis],axis=0).T))) # T
    
    # derivative with respect to Backscatter ratio exponent
    dNdB = np.diag(K*np.exp(phi)*Ts[inu0,:]*(Ts[inu0,:]*Trx[inu0]/BSR**2 \
        -1/BSR**2*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0)))*np.exp(lnBSR)[:,np.newaxis]
#        
    # derivative definitions:
    #   range axis - columns
    #   state parameter axis - rows
    return N,dNdB,dNdnWV,dNdT
    

def O2DIALDerivative(lnT,lnnWV,lnBSR,phi,rb_spec,abs_spec,Trx,inu0,K,base_T,base_P,dr,r0):
    
    BSR = np.exp(lnBSR)+1
#    T = np.exp(lnT)
    T = lnT
    nWV = np.exp(lnnWV)    
    
    # RB sensitivity needed for temperature derivative    
    betaM,dbetaMdT = calc_pca_spectrum_w_deriv(rb_spec,T,base_T,base_P)
    
    # estimate pressure from base station
    convP = 101325.0  # pressure conversion factor for O2 number density calculation
    P = convP*base_P*(T/base_T)**5.2199     
    base_Pc = convP*base_P  # converted base pressure from atm to Pa
    # *101325.0
    # Absorption sensitivity
    sigS,dsigSdT = calc_pca_spectrum_w_deriv(abs_spec,T,base_T,base_P)
    dr_int = np.ones((1,nWV.size))*dr
    dr_int[0,0] = r0
    ODs = np.cumsum(fo2*sigS*dr_int*(P/(lp.kB*T)-nWV)[np.newaxis,:],axis=1)
    Ts = np.exp(-ODs)
    
    # These definitions are specific to water vapor
#    dODsdnWV = np.cumsum(fo2*sigS,axis=1)*dr
#    dODsdT = np.cumsum(fo2*P[np.newaxis,:]/(lp.kB*T[np.newaxis,:])*(dsigSdT-sigS/T[np.newaxis,:]),axis=1)
    
    # These definitions are specific to oxygen
    # full 3D OD derivatives
    dODsdnWV = -np.cumsum((fo2*sigS*dr_int)[:,:,np.newaxis]*np.eye(T.size)[np.newaxis,:,:],axis=1)
    
#    dODsdT = np.cumsum((fo2*dsigSdT*(P*101325.0/(lp.kB*T)-nWV)[np.newaxis,:]-fo2*sigS*(P*101325.0/(lp.kB*T**2))[np.newaxis,:]) \
#        [:,:,np.newaxis]*np.eye(T.size)[np.newaxis,:,:],axis=1)*dr
        
    dODsdT = np.cumsum((fo2*dsigSdT*dr_int*(P/(lp.kB*T)-nWV)[np.newaxis,:]+fo2*sigS*dr_int*4.2199*(base_Pc*T**3.2199/(lp.kB*base_T**5.2199))[np.newaxis,:]) \
        [:,:,np.newaxis]*np.eye(T.size)[np.newaxis,:,:],axis=1)
    
    # laser line OD derivatives
    dODsdnWV_nuL = np.cumsum(np.diag(fo2*sigS[inu0,:]*dr_int.flatten()),axis=1) 

    # removed negation 3/29/2018
#    dODsdT_nuL = np.cumsum(np.diag(fo2*dsigSdT[inu0,:]*(P*101325.0/(lp.kB*T)-nWV)-fo2*sigS[inu0,:]*P*101325.0/(lp.kB*T**2)),axis=1)*dr
    dODsdT_nuL = -np.cumsum(np.diag(fo2*dsigSdT[inu0,:]*dr_int.flatten()*(P/(lp.kB*T)-nWV)+fo2*sigS[inu0,:]*dr_int.flatten()*4.2199*base_Pc*T**3.2199/(lp.kB*base_T**5.2199)),axis=1)
    
    # Calculate O2 profile
    N = K*np.exp(phi)*Ts[inu0,:]*(Ts[inu0,:]*Trx[inu0]*(1-1/BSR) \
        +1/BSR*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0))
      
      
    # derivative with respect to water vapor exponent
    # full 3D matrix integral
    dNdnWV = (N[np.newaxis,:]*dODsdnWV_nuL \
        +(K*np.exp(phi)*Ts[inu0,:])[np.newaxis,:]*((Ts[inu0,:])[np.newaxis,:]*dODsdnWV_nuL*Trx[inu0]*(1-1/BSR[np.newaxis,:]) \
        -1/BSR[np.newaxis,:]*np.sum(Trx[:,np.newaxis,np.newaxis]*Ts[:,:,np.newaxis]*dODsdnWV*betaM[:,:,np.newaxis],axis=0).T))*nWV[:,np.newaxis]
        
    # derivative with respect to temperature exponent
    # full 3D matrix integral
#    dNdT = (N[np.newaxis,:]*dODsdT_nuL \
#        +(K*np.exp(phi)*Ts[inu0,:])[np.newaxis,:]*((Ts[inu0,:])[np.newaxis,:]*dODsdT_nuL*Trx[inu0]*(1-1/BSR[np.newaxis,:]) \
#        +1/BSR[np.newaxis,:]*(np.diag(np.sum(Trx[:,np.newaxis]*Ts*dbetaMdT,axis=0)) \
#        -np.sum(Trx[:,np.newaxis,np.newaxis]*Ts[:,:,np.newaxis]*dODsdT*betaM[:,:,np.newaxis],axis=0).T)))*T[:,np.newaxis] # T exponent
    dNdT = (N[np.newaxis,:]*dODsdT_nuL \
        +(K*np.exp(phi)*Ts[inu0,:])[np.newaxis,:]*((Ts[inu0,:])[np.newaxis,:]*dODsdT_nuL*Trx[inu0]*(1-1/BSR[np.newaxis,:]) \
        +1/BSR[np.newaxis,:]*(np.diag(np.sum(Trx[:,np.newaxis]*Ts*dbetaMdT,axis=0)) \
        -np.sum(Trx[:,np.newaxis,np.newaxis]*Ts[:,:,np.newaxis]*dODsdT*betaM[:,:,np.newaxis],axis=0).T))) # T
    
    # derivative with respect to backscatter ratio exponent
    dNdB = np.diag(K*np.exp(phi)*Ts[inu0,:]*(Ts[inu0,:]*Trx[inu0]/BSR**2 \
        -1/BSR**2*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0)))*np.exp(lnBSR)[:,np.newaxis]
        
        
    return N,dNdB,dNdnWV,dNdT
  





    
    
def HSRLProfile(lnT,lnBSR,phi,rb_spec,Trx,inu0,K,base_T,base_P):
    
    BSR = np.exp(lnBSR)+1
#    T = np.exp(lnT)
    T = lnT
    # RB sensitivity needed for temperature derivative    
    betaM = calc_pca_spectrum(rb_spec,T,base_T,base_P) 
#    normbetaM = np.sum(betaM,axis=0)
#    betaM = betaM/normbetaM[np.newaxis]
    
    # HSRL Profile
    N = K*np.exp(phi)*(Trx[inu0]*(1-1/BSR)+1/BSR*np.sum(Trx[:,np.newaxis]*betaM,axis=0))
    
    return N

def WVDIALProfile(lnT,lnnWV,lnBSR,phi,rb_spec,abs_spec,Trx,inu0,K,base_T,base_P,dr,r0):
    
    BSR = np.exp(lnBSR)+1
#    T = np.exp(lnT)
    T = lnT
    nWV = np.exp(lnnWV)
    
    # RB sensitivity needed for temperature derivative    
    betaM = calc_pca_spectrum(rb_spec,T,base_T,base_P)
#    normbetaM = np.sum(betaM,axis=0)
#    betaM = betaM/normbetaM[np.newaxis]
    
    # Absorption sensitivity
    sigS = calc_pca_spectrum(abs_spec,T,base_T,base_P)
    dr_int = np.ones((1,nWV.size))*dr
#    plt.figure(); plt.plot(dr_int.flatten())
    dr_int[0,0] = r0
    ODs = np.cumsum(nWV*sigS*dr_int,axis=1)
    Ts = np.exp(-ODs)
    
    # Calculate WV profile
    N = K*np.exp(phi)*Ts[inu0,:]*(Ts[inu0,:]*Trx[inu0]*(1-1/BSR) \
        +1/BSR*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0))
                
    return N
    

def O2DIALProfile(lnT,lnnWV,lnBSR,phi,rb_spec,abs_spec,Trx,inu0,K,base_T,base_P,dr,r0):
    
    BSR = np.exp(lnBSR)+1
#    T = np.exp(lnT)
    T = lnT
    nWV = np.exp(lnnWV)
    
    # RB sensitivity needed for temperature derivative    
    betaM = calc_pca_spectrum(rb_spec,T,base_T,base_P) 
#    normbetaM = np.sum(betaM,axis=0)
#    betaM = betaM/normbetaM[np.newaxis]
    
    # estimate pressure from base station
    convP = 101325.0  # pressure conversion factor for O2 number density calculation
    P = convP*base_P*(T/base_T)**5.2199     
    
    # Absorption sensitivity
    dr_int = np.ones((1,nWV.size))*dr
    dr_int[0,0] = r0
    sigS = calc_pca_spectrum(abs_spec,T,base_T,base_P)
    ODs = np.cumsum(dr_int*fo2*sigS*(P/(lp.kB*T)-nWV)[np.newaxis,:],axis=1)
    Ts = np.exp(-ODs)
    
    # Calculate WV profile
    N = K*np.exp(phi)*Ts[inu0,:]*(Ts[inu0,:]*Trx[inu0]*(1-1/BSR) \
        +1/BSR*np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0))
        
        
    return N

"""
The below functions calc_pca_spectrum and calc_pca_spectrum_w_deriv 
are different from the versions in SpectrumLib.  These versions estimate
pressure based on a base station T and P and ideal gas law.  The SpectrumLib
versions accept pressure as an independent input.

The link between T and P is important in the lidar optimization problem where
modifying the temperature will also modify the pressure.
"""
  
  
def calc_pca_spectrum(spec_data,T,base_T,base_P,nu=np.array([])):
    """
    Calculates extinction or Rayleigh Brillioun spectrum for 
    T and P arrays (of equal size)
    spec_data - dict of PCA spectral data returned by load_spect_params
    if nu is unassigned, the data is returned at native resolution
        (this is the fasted option)
    if nu is defined, the data is interpolated to the requested grid
    
    T is expected in K
    P is expected in atm
    
    If speed is needed it is not recomended that nu be provided.  Instead
    provide the required frequency grid to load_spect_params.  That will
    configure the PCA matrices so no interpolation is required.
    """
    
    P = base_P*(T.flatten()/base_T)**5.2199    
    
    TPmat = np.matrix(((T.flatten()-spec_data['Tmean'])/spec_data['Tstd'])[:,np.newaxis]**spec_data['pT'] \
        *((P.flatten()-spec_data['Pmean'])/spec_data['Pstd'])[:,np.newaxis]**spec_data['pP'])
    extinction = np.array(spec_data['M']*TPmat.T+spec_data['Mavg'])

    if nu.size > 0:
        # if nu is provided, interpolate to obtain requrested frequency grid
        ext_interp = np.zeros((nu.size,T.size))
   
        for ai in range(T.size):
            ext_interp[:,ai] = np.interp(nu,spec_data['nu_pca'],extinction[:,ai],left=0,right=0)
        return ext_interp   
    else:
        return extinction
        
def calc_pca_spectrum_w_deriv(spec_data,T,base_T,base_P,nu=np.array([])):
    """
    Calculates extinction spectrum and its temperature derivative for T and P 
    arrays (of equal size)
    spec_data - dict of PCA spectral data returned by load_spect_params
    if nu is unassigned, the data is returned at native resolution
        (this is the fasted option)
    if nu is defined, the data is interpolated to the requested grid
    
    T is expected in K
    P is expected in atm
    
    If speed is needed it is not recomended that nu be provided.  Instead
    provide the required frequency grid to load_spect_params.  That will
    configure the PCA matrices so no interpolation is required.
    """
    
    P = base_P*(T.flatten()/base_T)**5.2199 
    
    TPmat = np.matrix(((T.flatten()-spec_data['Tmean'])/spec_data['Tstd'])[:,np.newaxis]**spec_data['pT'] \
        *((P.flatten()-spec_data['Pmean'])/spec_data['Pstd'])[:,np.newaxis]**spec_data['pP'])
    
    # temperature derivative of theta including pressure dependence on this term
    dTPmat = np.matrix(spec_data['pT']/spec_data['Tstd']*(((T.flatten()-spec_data['Tmean'])/spec_data['Tstd'])[:,np.newaxis]**(spec_data['dpT'])) \
        *(((P.flatten()-spec_data['Pmean'])/spec_data['Pstd'])[:,np.newaxis]**spec_data['pP'])) \
        + np.matrix(spec_data['pP']/spec_data['Pstd']*(5.2199*base_P/base_T)*(T.flatten()[:,np.newaxis]/base_T)**(4.2199) \
        *(((T.flatten()-spec_data['Tmean'])/spec_data['Tstd'])[:,np.newaxis]**(spec_data['pT'])) \
        *(((P.flatten()-spec_data['Pmean'])/spec_data['Pstd'])[:,np.newaxis]**(spec_data['dpP'])))

    extinction = np.array(spec_data['M']*TPmat.T+spec_data['Mavg'])
    dextinction = np.array(spec_data['M']*dTPmat.T)

    if nu.size > 0:
        # if nu is provided, interpolate to obtain requrested frequency grid
        ext_interp = np.zeros((nu.size,T.size))
        dext_interp = np.zeros((nu.size,T.size))
   
        for ai in range(T.size):
            ext_interp[:,ai] = np.interp(nu,spec_data['nu_pca'],extinction[:,ai],left=0,right=0)
            dext_interp[:,ai] = np.interp(nu,spec_data['nu_pca'],dextinction[:,ai],left=0,right=0)
        return ext_interp,dext_interp
    else:
        return extinction,dextinction