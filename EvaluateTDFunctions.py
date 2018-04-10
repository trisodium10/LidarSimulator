# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:16:28 2017

@author: mhayman
"""

import numpy as np
import LidarProfileFunctions as lp
#import WVProfileFunctions as wv
import FourierOpticsLib as FO

import TDRetrievalLib as td
import TDRetrieval2Lib as td2
import TDRetrieval3Lib as td3

import SpectrumLib as spec

import matplotlib.pyplot as plt

"""
Analysis definitions
"""

a_wavelength_list = [828.3026e-9,828.203e-9,780.246119e-9,769.2339e-9,769.319768e-9]  # offline,online,Comb,Mol,online,offline  # O2 On center: 769.2339e-9
a_name_list = ['WV Offline','WV Online','HSRL','O2 Online','O2 Offline']  # name corresponding to each wavelength

a_dnu = 20e6  # spacing in optical frequency space of simulation
a_nu_max = 4e9  # edge of simulated frequency space



wl = dict(zip(a_name_list,a_wavelength_list))  # dictionary of wavelengths

#sim_range = np.arange(0,15e3,sim_dr)  # range array in m
nu = np.arange(-a_nu_max,a_nu_max,a_dnu) # frequency array in Hz
inu0 = np.argmin(np.abs(nu)) # index to center of frequency array

# Load PCA spectroscopy data
# HSRL
hsrl_rb_fn = spec.get_pca_filename('RB',name='HSRL')
hsrl_rb = spec.load_spect_params(hsrl_rb_fn,nu=nu)

# WV-DIAL
wv_rb_on_fn = spec.get_pca_filename('RB',name='WV Online')
wv_rb_on = spec.load_spect_params(wv_rb_on_fn,nu=nu)
wv_abs_on_fn = spec.get_pca_filename('Abs',name='WV Online')
wv_abs_on = spec.load_spect_params(wv_abs_on_fn,nu=nu)

wv_rb_off_fn = spec.get_pca_filename('RB',name='WV Offline')
wv_rb_off = spec.load_spect_params(wv_rb_off_fn,nu=nu)
wv_abs_off_fn = spec.get_pca_filename('Abs',name='WV Offline')
wv_abs_off = spec.load_spect_params(wv_abs_off_fn,nu=nu)

# O2-DIAL
o2_rb_on_fn = spec.get_pca_filename('RB',name='O2 Online')
o2_rb_on = spec.load_spect_params(o2_rb_on_fn,nu=nu)
o2_abs_on_fn = spec.get_pca_filename('Abs',name='O2 Online')
o2_abs_on = spec.load_spect_params(o2_abs_on_fn,nu=nu)

o2_rb_off_fn = spec.get_pca_filename('RB',name='O2 Offline')
o2_rb_off = spec.load_spect_params(o2_rb_off_fn,nu=nu)
o2_abs_off_fn = spec.get_pca_filename('Abs',name='O2 Offline')
o2_abs_off = spec.load_spect_params(o2_abs_off_fn,nu=nu)

# build spectrscopy dictionaries
rb_spec = dict(zip(a_name_list,[wv_rb_off,wv_rb_on,hsrl_rb,o2_rb_on,o2_rb_off]))
abs_spec = dict(zip(a_name_list[0:2]+a_name_list[3:5],[wv_abs_off,wv_abs_on,o2_abs_on,o2_abs_off]))

Etalon_angle = 0.00*np.pi/180
WV_Filter = FO.FP_Etalon(1.0932e9,43.5e9,lp.c/wl['WV Online'],efficiency=0.95,InWavelength=False)

HSRL_Etalon_angle = 0.00*np.pi/180
HSRL_Filter1 = FO.FP_Etalon(5.7e9,45e9,lp.c/wl['HSRL'],efficiency=0.95,InWavelength=False)
HSRL_Filter2 = FO.FP_Etalon(15e9,250e9,lp.c/wl['HSRL'],efficiency=0.95,InWavelength=False)
HSRL_RbFilter = spec.RubidiumCellTransmission(nu+lp.c/wl['HSRL'],50+274.15,0,7.2e-2,iso87=0.97)

O2_Etalon_angle = 0.00*np.pi/180
O2_Filter = FO.FP_Etalon(1.0932e9,43.5e9,lp.c/wl['O2 Online'],efficiency=0.95,InWavelength=False)


# define receiver transmission function dictionary
Trx = {}
Trx['WV Online'] = WV_Filter.spectrum(nu+lp.c/wl['WV Online'],InWavelength=False)
Trx['WV Offline'] = WV_Filter.spectrum(nu+lp.c/wl['WV Offline'],InWavelength=False)
Trx['HSRL Mol'] = HSRL_Filter1.spectrum(nu+lp.c/wl['HSRL'],InWavelength=False) \
    *HSRL_Filter2.spectrum(nu+lp.c/wl['HSRL'],InWavelength=False) \
    *HSRL_RbFilter
Trx['HSRL Comb'] = HSRL_Filter1.spectrum(nu+lp.c/wl['HSRL'],InWavelength=False) \
    *HSRL_Filter2.spectrum(nu+lp.c/wl['HSRL'],InWavelength=False)
Trx['O2 Online'] = O2_Filter.spectrum(nu+lp.c/wl['O2 Online'],InWavelength=False)
Trx['O2 Offline'] = O2_Filter.spectrum(nu+lp.c/wl['O2 Offline'],InWavelength=False)

# define BSR multipliers
multBSR = {'WV':(wl['WV Online']/wl['HSRL'])**3, 'O2':(wl['O2 Online']/wl['HSRL'])**3}


"""
Simulation parameters
"""

sim_T0 = 320  # temp at surface in K
sim_dT = 9.3e-3  # lapse rate in K/m
TauP = 8e3  # pressure decay constant in m altitude
P0 = 1.0  # pressure at base station
sim_dr = 37.5  # simulated range resolution

sim_range = np.arange(0,8e3,sim_dr)  # range array in m
#sim_P = 1.0*np.exp(-sim_range/TauP)
sim_T = sim_T0-sim_dT*sim_range
sim_P = P0*(sim_T/sim_T[0])**5.2199

# absolute humidity initally in g/m^3
sim_nWVi = 20*(1-sim_range/6e3)
sim_nWVi[np.nonzero(sim_range > 4e3)] = 0.6
p_sim_wv = np.polyfit(sim_range,np.log(sim_nWVi),13)
sim_nWV = np.exp(np.polyval(p_sim_wv,sim_range))
sim_nWV = sim_nWV*lp.N_A/lp.mH2O  # convert to number/m^3

# aerosol backscatter coefficient m^-1 sr^-1 at 780 nm
beta_aer_hsrl = 1e-8*np.ones(sim_range.size)
beta_aer_hsrl[np.nonzero(sim_range < 2e3)] = 1e-6
p_sim_aer = np.polyfit(sim_range,np.log10(beta_aer_hsrl),13)
sim_beta_aer = 10**(np.polyval(p_sim_aer,sim_range))
beta_mol = 5.45*(550.0e-9/wl['WV Online'])**4*1e-32*(sim_P/9.86923e-6)/(sim_T*lp.kB)
sim_BSR = (sim_beta_aer+beta_mol)/beta_mol
wv_BSR = (sim_BSR-1)*multBSR['WV']+1
o2_BSR = (sim_BSR-1)*multBSR['O2']+1

"""
Evaluate Derivative Functions from td
"""

### Test WV-DIAL WV derivative
#dOD = td.WVDIALDerivative(sim_T,sim_P,sim_nWV,wv_BSR,rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0,1.0,sim_dr)
#Nwv = lambda x: td.WVDIALProfile(sim_T,sim_P,x,wv_BSR,rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0,1.0,sim_dr)
#dODWVnum = td.Num_Jacob(Nwv,sim_nWV,step_size=1e-9)
#plt.figure(); 
#plt.semilogy(np.sum(-dOD[2].T,axis=1)); 
#plt.semilogy(np.sum(-dODWVnum,axis=1),'--');
#plt.title('WV-DIAL, WV derivative')
#plt.legend(['Explicit','Numeric'])
#
## Test WV-DIAL T derivative
#NwvT = lambda x: td.WVDIALProfile(x,sim_P,sim_nWV,wv_BSR,rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0,1.0,sim_dr)
#dODTnum = td.Num_Jacob(NwvT,sim_T,step_size=1e-9)
#plt.figure(); 
#plt.semilogy(np.nansum(dOD[3].T,axis=1)); 
#plt.semilogy(np.nansum(dODTnum,axis=1),'--');
#plt.title('WV-DIAL, T derivative')
#plt.legend(['Explicit','Numeric'])
#
## Test WV-DIAL BSR derivative
#NwvB = lambda x: td.WVDIALProfile(sim_T,sim_P,sim_nWV,x,rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0,1.0,sim_dr)
#dODBnum = td.Num_Jacob(NwvB,wv_BSR,step_size=1e-9)
#plt.figure(); 
#plt.semilogy(np.nansum(dOD[1].T,axis=1)); 
#plt.semilogy(np.nansum(dODBnum,axis=1),'--');
#plt.title('WV-DIAL, BSR derivative')
#plt.legend(['Explicit','Numeric'])
#
#
## Test O2-DIAL WV derivative
#dODo = td.O2DIALDerivative(sim_T,sim_P,sim_nWV,o2_BSR,rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0,1.0,sim_dr)
#No2 = lambda x: td.O2DIALProfile(sim_T,sim_P,x,o2_BSR,rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0,1.0,sim_dr)
#dODoWVnum = td.Num_Jacob(No2,sim_nWV,step_size=1e-9)
#plt.figure(); 
#plt.semilogy(np.sum(dODo[2].T,axis=1)); 
#plt.semilogy(np.sum(dODoWVnum,axis=1),'--');
#plt.title('O2-DIAL, WV derivative')
#plt.legend(['Explicit','Numeric'])
#
## Test O2-DIAL T derivative
#No2T = lambda x: td.O2DIALProfile(x,sim_P,sim_nWV,o2_BSR,rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0,1.0,sim_dr)
#dODoTnum = td.Num_Jacob(No2T,sim_T,step_size=1e-9)
#plt.figure(); 
#plt.plot(np.nansum(dODo[3].T,axis=1)); 
#plt.plot(np.nansum(dODoTnum,axis=1),'--');
#plt.title('O2-DIAL, T derivative')
#plt.legend(['Explicit','Numeric'])
#
## Test O2-DIAL BSR derivative
#No2B = lambda x: td.O2DIALProfile(sim_T,sim_P,sim_nWV,x,rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0,1.0,sim_dr)
#dODoBnum = td.Num_Jacob(No2B,o2_BSR,step_size=1e-9)
#plt.figure(); 
#plt.semilogy(np.nansum(dODo[1].T,axis=1)); 
#plt.semilogy(np.nansum(dODoBnum,axis=1),'--');
#plt.title('O2-DIAL, BSR derivative')
#plt.legend(['Explicit','Numeric'])
#
#
## Test HSRL T derivative
#dODh = td.HSRLDerivative(sim_T,sim_P,sim_BSR,rb_spec['HSRL'],Trx['HSRL Mol'],inu0,1.0)
#Nh2T = lambda x: td.HSRLProfile(x,sim_P,sim_BSR,rb_spec['HSRL'],Trx['HSRL Mol'],inu0,1.0)
#dODhTnum = td.Num_Jacob(Nh2T,sim_T,step_size=1e-9)
#plt.figure(); 
#plt.plot(np.nansum(dODh[2].T,axis=1)); 
#plt.plot(np.nansum(dODhTnum,axis=1),'--');
#plt.title('HSRL, T derivative')
#plt.legend(['Explicit','Numeric'])
#
## Test HSRL BSR derivative
#No2B = lambda x: td.HSRLProfile(sim_T,sim_P,x,rb_spec['HSRL'],Trx['HSRL Mol'],inu0,1.0)
#dODhBnum = td.Num_Jacob(No2B,sim_BSR,step_size=1e-9)
#plt.figure(); 
#plt.semilogy(np.nansum(-dODh[1].T,axis=1)); 
#plt.semilogy(np.nansum(-dODhBnum,axis=1),'--');
#plt.title('HSRL, BSR derivative')
#plt.legend(['Explicit','Numeric'])



"""
Evaluate Derivative Functions from td2
"""

#test_phi = np.zeros(sim_nWV.shape)  # common term needed for td2 functions
#
### Test WV-DIAL WV derivative
#dOD = td2.WVDIALDerivative(sim_T,sim_nWV,wv_BSR,test_phi,rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0,1.0,sim_T[0],P0,sim_dr)
#Nwv = lambda x: td2.WVDIALProfile(sim_T,x,wv_BSR,test_phi,rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0,1.0,sim_T[0],P0,sim_dr)
#dODWVnum = td2.Num_Jacob(Nwv,sim_nWV,step_size=1e-9)
#plt.figure(); 
#plt.semilogy(np.sum(-dOD[2].T,axis=1)); 
#plt.semilogy(np.sum(-dODWVnum,axis=1),'--');
#plt.title('TD2: WV-DIAL, WV derivative')
#plt.legend(['Explicit','Numeric'])
#
## Test WV-DIAL T derivative
#NwvT = lambda x: td2.WVDIALProfile(x,sim_nWV,wv_BSR,test_phi,rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0,1.0,sim_T[0],P0,sim_dr)
#dODTnum = td2.Num_Jacob(NwvT,sim_T,step_size=1e-9)
#plt.figure(); 
#plt.plot(np.nansum(dOD[3].T,axis=1)); 
#plt.plot(np.nansum(dODTnum,axis=1),'--');
#plt.title('TD2: WV-DIAL, T derivative')
#plt.legend(['Explicit','Numeric'])
#
## Test WV-DIAL BSR derivative
#NwvB = lambda x: td2.WVDIALProfile(sim_T,sim_nWV,x,test_phi,rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0,1.0,sim_T[0],P0,sim_dr)
#dODBnum = td2.Num_Jacob(NwvB,wv_BSR,step_size=1e-9)
#plt.figure(); 
#plt.semilogy(np.nansum(dOD[1].T,axis=1)); 
#plt.semilogy(np.nansum(dODBnum,axis=1),'--');
#plt.title('TD2: WV-DIAL, BSR derivative')
#plt.legend(['Explicit','Numeric'])
#
#
## Test O2-DIAL WV derivative
#dODo = td2.O2DIALDerivative(sim_T,sim_nWV,o2_BSR,test_phi,rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0,1.0,sim_T[0],P0,sim_dr)
#No2 = lambda x: td2.O2DIALProfile(sim_T,x,o2_BSR,test_phi,rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0,1.0,sim_T[0],P0,sim_dr)
#dODoWVnum = td2.Num_Jacob(No2,sim_nWV,step_size=1e-9)
#plt.figure(); 
#plt.semilogy(np.sum(dODo[2].T,axis=1)); 
#plt.semilogy(np.sum(dODoWVnum,axis=1),'--');
#plt.title('TD2: O2-DIAL, WV derivative')
#plt.legend(['Explicit','Numeric'])
#
## Test O2-DIAL T derivative
#No2T = lambda x: td2.O2DIALProfile(x,sim_nWV,o2_BSR,test_phi,rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0,1.0,sim_T[0],P0,sim_dr)
#dODoTnum = td2.Num_Jacob(No2T,sim_T,step_size=1e-9)
#plt.figure(); 
#plt.plot(np.nansum(dODo[3].T,axis=1)); 
#plt.plot(np.nansum(dODoTnum,axis=1),'--');
#plt.title('TD2: O2-DIAL, T derivative')
#plt.legend(['Explicit','Numeric'])
#
## Test O2-DIAL BSR derivative
#No2B = lambda x: td2.O2DIALProfile(sim_T,sim_nWV,x,test_phi,rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0,1.0,sim_T[0],P0,sim_dr)
#dODoBnum = td2.Num_Jacob(No2B,o2_BSR,step_size=1e-9)
#plt.figure(); 
#plt.semilogy(np.nansum(dODo[1].T,axis=1)); 
#plt.semilogy(np.nansum(dODoBnum,axis=1),'--');
#plt.title('TD2: O2-DIAL, BSR derivative')
#plt.legend(['Explicit','Numeric'])
#
#
## Test HSRL T derivative
#dODh = td2.HSRLDerivative(sim_T,sim_BSR,test_phi,rb_spec['HSRL'],Trx['HSRL Mol'],inu0,1.0,sim_T[0],P0)
#Nh2T = lambda x: td2.HSRLProfile(x,sim_BSR,test_phi,rb_spec['HSRL'],Trx['HSRL Mol'],inu0,1.0,sim_T[0],P0)
#dODhTnum = td.Num_Jacob(Nh2T,sim_T,step_size=1e-9)
#plt.figure(); 
#plt.plot(np.nansum(dODh[2].T,axis=1)); 
#plt.plot(np.nansum(dODhTnum,axis=1),'--');
#plt.title('TD2: HSRL, T derivative')
#plt.legend(['Explicit','Numeric'])
#
## Test HSRL BSR derivative
#No2B = lambda x: td2.HSRLProfile(sim_T,x,test_phi,rb_spec['HSRL'],Trx['HSRL Mol'],inu0,1.0,sim_T[0],P0)
#dODhBnum = td2.Num_Jacob(No2B,sim_BSR,step_size=1e-9)
#plt.figure(); 
#plt.semilogy(np.nansum(-dODh[1].T,axis=1)); 
#plt.semilogy(np.nansum(-dODhBnum,axis=1),'--');
#plt.title('TD2: HSRL, BSR derivative')
#plt.legend(['Explicit','Numeric'])


"""
Evaluate Derivative Functions from td3 - exponent state variables
"""


irange = 10
test_T = sim_T[irange:]
test_nWV = sim_nWV[irange:]
test_wvBSR = wv_BSR[irange:]
test_o2BSR = o2_BSR[irange:]
test_BSR = sim_BSR[irange:]
test_phi = np.zeros(test_nWV.shape)  # common term needed for td3 functions

## Test WV-DIAL WV derivative
dOD = td3.WVDIALDerivative(np.log(test_T),np.log(test_nWV),np.log(test_wvBSR-1),test_phi,rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0,1.0,sim_T[0],P0,sim_dr,sim_range[irange])
Nwv = lambda x: td3.WVDIALProfile(np.log(test_T),x,np.log(test_wvBSR-1),test_phi,rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0,1.0,sim_T[0],P0,sim_dr,sim_range[irange])
dODWVnum = td3.Num_Jacob(Nwv,np.log(test_nWV),step_size=1e-9)
plt.figure(); 
plt.plot(np.sum(dOD[2].T,axis=1)); 
plt.plot(np.sum(dODWVnum,axis=1),'--');
plt.title('TD3: WV-DIAL, WV derivative')
plt.legend(['Explicit','Numeric'])

# Test WV-DIAL T derivative
NwvT = lambda x: td3.WVDIALProfile(x,np.log(test_nWV),np.log(test_wvBSR-1),test_phi,rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0,1.0,sim_T[0],P0,sim_dr,sim_range[irange])
dODTnum = td3.Num_Jacob(NwvT,np.log(test_T),step_size=1e-9)
plt.figure(); 
plt.plot(np.nansum(dOD[3].T,axis=1)); 
plt.plot(np.nansum(dODTnum,axis=1),'--');
plt.title('TD3: WV-DIAL, T derivative')
plt.legend(['Explicit','Numeric'])

# Test WV-DIAL BSR derivative
NwvB = lambda x: td3.WVDIALProfile(np.log(test_T),np.log(test_nWV),x,test_phi,rb_spec['WV Online'],abs_spec['WV Online'],Trx['WV Online'],inu0,1.0,sim_T[0],P0,sim_dr,sim_range[irange])
dODBnum = td3.Num_Jacob(NwvB,np.log(test_wvBSR-1),step_size=1e-9)
plt.figure(); 
plt.plot(np.nansum(dOD[1].T,axis=1)); 
plt.plot(np.nansum(dODBnum,axis=1),'--');
plt.title('TD3: WV-DIAL, BSR derivative')
plt.legend(['Explicit','Numeric'])


# Test O2-DIAL WV derivative
dODo = td3.O2DIALDerivative(np.log(test_T),np.log(test_nWV),np.log(test_o2BSR-1),test_phi,rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0,1.0,sim_T[0],P0,sim_dr,sim_range[irange])
No2 = lambda x: td3.O2DIALProfile(np.log(test_T),x,np.log(test_o2BSR-1),test_phi,rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0,1.0,sim_T[0],P0,sim_dr,sim_range[irange])
dODoWVnum = td3.Num_Jacob(No2,np.log(test_nWV),step_size=1e-9)
plt.figure(); 
plt.plot(np.sum(dODo[2].T,axis=1)); 
plt.plot(np.sum(dODoWVnum,axis=1),'--');
plt.title('TD3: O2-DIAL, WV derivative')
plt.legend(['Explicit','Numeric'])

# Test O2-DIAL T derivative
No2T = lambda x: td3.O2DIALProfile(x,np.log(test_nWV),np.log(test_o2BSR-1),test_phi,rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0,1.0,sim_T[0],P0,sim_dr,sim_range[irange])
dODoTnum = td3.Num_Jacob(No2T,np.log(test_T),step_size=1e-9)
plt.figure(); 
plt.plot(np.nansum(dODo[3].T,axis=1)); 
plt.plot(np.nansum(dODoTnum,axis=1),'--');
plt.title('TD3: O2-DIAL, T derivative')
plt.legend(['Explicit','Numeric'])

# Test O2-DIAL BSR derivative
No2B = lambda x: td3.O2DIALProfile(np.log(test_T),np.log(test_nWV),x,test_phi,rb_spec['O2 Online'],abs_spec['O2 Online'],Trx['O2 Online'],inu0,1.0,sim_T[0],P0,sim_dr,sim_range[irange])
dODoBnum = td3.Num_Jacob(No2B,np.log(test_o2BSR-1),step_size=1e-9)
plt.figure(); 
plt.plot(np.nansum(dODo[1].T,axis=1)); 
plt.plot(np.nansum(dODoBnum,axis=1),'--');
plt.title('TD3: O2-DIAL, BSR derivative')
plt.legend(['Explicit','Numeric'])


# Test HSRL T derivative
dODh = td3.HSRLDerivative(np.log(test_T),np.log(test_BSR-1),test_phi,rb_spec['HSRL'],Trx['HSRL Mol'],inu0,1.0,sim_T[0],P0)
Nh2T = lambda x: td3.HSRLProfile(x,np.log(test_BSR-1),test_phi,rb_spec['HSRL'],Trx['HSRL Mol'],inu0,1.0,sim_T[0],P0)
dODhTnum = td3.Num_Jacob(Nh2T,np.log(test_T),step_size=1e-9)
plt.figure(); 
plt.plot(np.nansum(dODh[2].T,axis=1)); 
plt.plot(np.nansum(dODhTnum,axis=1),'--');
plt.title('TD3: HSRL, T derivative')
plt.legend(['Explicit','Numeric'])

# Test HSRL BSR derivative
No2B = lambda x: td3.HSRLProfile(np.log(test_T),x,test_phi,rb_spec['HSRL'],Trx['HSRL Mol'],inu0,1.0,sim_T[0],P0)
dODhBnum = td3.Num_Jacob(No2B,np.log(test_BSR-1),step_size=1e-9)
plt.figure(); 
plt.plot(np.nansum(dODh[1].T,axis=1)); 
plt.plot(np.nansum(dODhBnum,axis=1),'--');
plt.title('TD3: HSRL, BSR derivative')
plt.legend(['Explicit','Numeric'])



T0 = np.array([200])
var = 'WV Online'
sigS,dsigSdT = td3.calc_pca_spectrum_w_deriv(abs_spec[var],T0,312,0.9)
NsigS = lambda x: td3.calc_pca_spectrum(abs_spec[var],x,312,0.9)
dsigSdTnum = td3.Num_Jacob(NsigS,T0,step_size=1e-6)
plt.figure()
plt.plot(dsigSdT)
plt.plot(dsigSdTnum,'--')
plt.title(var+' Temperature derivative')

plt.figure()
plt.plot(sigS)
plt.plot(NsigS(T0),'--')
plt.title(var+' Spectrum')