# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 13:32:33 2018

@author: mhayman
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
#from scipy.io import netcdf
import LidarProfileFunctions as lp
import LidarPlotFunctions as lplt
import WVProfileFunctions as wv
import FourierOpticsLib as FO
import SpectrumLib as spec

import sys

sys.path.append('/Users/mhayman/Documents/Python/Lidar')

import RandomVarLib as rv

import TDRetrieval3Lib as td # use the TD3 library that treats state variables as exponents (e.g. ln(T) = x)

import datetime

#import netCDF4 as nc4


#ncfile = 'simulated_thermodynamic_DIAL_20180726_T0859_'  # 2 minute data with shot noise
#ncfile = 'simulated_thermodynamic_DIAL_20180726_T1030_'  # 10 minute data with shot noise
#ncfile = 'simulated_thermodynamic_DIAL_20180727_T1408_'  # 2 minute data with shot noise without cloud
#ncfile = 'simulated_thermodynamic_DIAL_20180727_T1419_'  # 2 minute data with shot noise and elevated aerosol
ncfile = 'simulated_thermodynamic_DIAL_20180727_T1430_'  # 10 minute data with shot noise and elevated aerosol

load_sim_list = ['sim_T','sim_P','sim_range','sim_nu','Tetalon_O2_Online', 
    'Tetalon_O2_Offline','sim_nWV','BSR_HSRL_Molecular','Taer_HSRL_Molecular',
    'Overlap','sim_beta_mol','RbFilter','Tetalon_HSRL_Molecular',
    'Tetalon_WV_Online','Tetalon_WV_Offline','sim_nO2','sim_dr',
    'T_tx_O2_Online','T_rx_aer_O2_Online','T_rx_mol_O2_Online',
    'WV_Online_laser_spectrum','laser_bandwidth_WV_Online']
    
sim_vars = lp.load_nc_vars(ncfile+'sim.nc',load_sim_list)



load_profile_list = ['WV Offline','WV Online','HSRL Combined','HSRL Molecular','O2 Online','O2 Offline']
load_var_list = ['T_WS','P_WS']

profs = {}
for prof_name in load_profile_list:
    profs[prof_name] = lp.load_nc_Profile(ncfile+'data.nc',('Simulated ' + prof_name).replace(' ','_'))
    
data_vars = lp.load_nc_vars(ncfile+'data.nc',load_var_list)

"""
Process Data (Standard routines)
"""

lp.plotprofiles(profs)
raw_profs = {}
p_profs = {}
for var in profs.keys():
    raw_profs[var] = profs[var].copy()
    p_profs[var] = rv.load_array(profs[var].profile)
    
    profs[var].bg_subtract(-50)
    p_profs[var] = p_profs[var] - 1.0/50*np.sum(p_profs[var][:,-50:],axis=1)[:,np.newaxis]
#    p_profs[var] = p_profs[var] - 1.0/50*rv.varsum(p_profs[:,-50:],axis=1)

lp.plotprofiles(profs)

pltvar = 'WV Offline'
plt.figure()
plt.fill_betweenx(profs[pltvar].range_array,
                  profs[pltvar].profile[0,:]-np.sqrt(profs[pltvar].profile_variance[0,:]), \
                  profs[pltvar].profile[0,:]+np.sqrt(profs[pltvar].profile_variance[0,:]), \
                  color='r',alpha=0.2,label='First Order')
plt.fill_betweenx(profs[pltvar].range_array, \
                  rv.moment(p_profs[pltvar][0,:],num=1)-np.sqrt(rv.moment(p_profs[pltvar][0,:],num=2)), \
                  rv.moment(p_profs[pltvar][0,:],num=1)+np.sqrt(rv.moment(p_profs[pltvar][0,:],num=2)),color='b',alpha=0.2,label='Third Order')
plt.plot(profs[pltvar].profile[0,:],profs[pltvar].range_array,'b-')
plt.plot(rv.moment(p_profs[pltvar][0,:],num=0),profs[pltvar].range_array,'r-')
plt.grid(b=True)
plt.xlabel('Signal Counts')
plt.ylabel('Range [m]')

beta_mol_sonde,temp,pres = lp.get_beta_m_model(profs['HSRL Molecular'],np.array([data_vars['T_WS']]),np.array([data_vars['P_WS']]),returnTP=True)
#beta_mol_sonde,temp,pres = lp.get_beta_m_model(profs['HSRL Molecular'],data_vars['T_WS'],data_vars['P_WS'],returnTP=True)
pres.gain_scale(9.86923e-6)  
pres.profile_type = '$atm.$'



# etalon design
Etalon_angle = 0.00*np.pi/180
WV_Filter = FO.FP_Etalon(1.0932e9,43.5e9,lp.c/profs['WV Offline'].wavelength,efficiency=0.95,InWavelength=False)

HSRL_Etalon_angle = 0.00*np.pi/180
HSRL_Filter1 = FO.FP_Etalon(5.7e9,45e9,lp.c/profs['HSRL Molecular'].wavelength,efficiency=0.95,InWavelength=False)
HSRL_Filter2 = FO.FP_Etalon(15e9,250e9,lp.c/profs['HSRL Molecular'].wavelength,efficiency=0.95,InWavelength=False)

O2_Etalon_angle = 0.00*np.pi/180
O2_Filter = FO.FP_Etalon(1.0932e9,43.5e9,lp.c/profs['O2 Online'].wavelength,efficiency=0.95,InWavelength=False)



"""
Analysis definitions
"""
#a_wavelength_list = [wavelength_list[0],wavelength_list[1],wavelength_list[2],wavelength_list[4],wavelength_list[5]]
##a_wavelength_list = [828.3026e-9,828.203e-9,780.246119e-9,769.2365e-9+0.001e-9,769.319768e-9]  # offline,online,Comb,Mol,online,offline  # O2 On center: 769.2339e-9
#a_name_list = ['WV Offline','WV Online','HSRL','O2 Online','O2 Offline']  # name corresponding to each wavelength

range_limits = [500,6e3] # [500,6e3]

a_name_list = []
a_wavelength_list = []
for prof_name in profs.keys():
    a_wavelength_list.extend([profs[prof_name].wavelength])
    a_name_list.extend([str(prof_name)])


a_dnu = 20e6  # spacing in optical frequency space of simulation # 20e6 original
a_nu_max = 4e9  # edge of simulated frequency space  

MolGain = 2.02  # estimated from BSR profile

wl = dict(zip(a_name_list,a_wavelength_list))  # dictionary of wavelengths

#sim_range = np.arange(0,15e3,sim_dr)  # range array in m
nu = np.arange(-a_nu_max,a_nu_max,a_dnu) # frequency array in Hz
inu0 = np.argmin(np.abs(nu)) # index to center of frequency array

# Load PCA spectroscopy data
# HSRL
hsrl_rb_fn = spec.get_pca_filename('RB',name='HSRL')
hsrl_rb = spec.load_spect_params(hsrl_rb_fn,nu=nu,normalize=True)

# WV-DIAL
wv_rb_on_fn = spec.get_pca_filename('RB',name='WV Online')
wv_rb_on = spec.load_spect_params(wv_rb_on_fn,nu=nu,normalize=True)
wv_abs_on_fn = spec.get_pca_filename('Abs',name='WV Online')
wv_abs_on = spec.load_spect_params(wv_abs_on_fn,nu=nu,wavelen=profs['WV Online'].wavelength)

wv_rb_off_fn = spec.get_pca_filename('RB',name='WV Offline')
wv_rb_off = spec.load_spect_params(wv_rb_off_fn,nu=nu,normalize=True)
wv_abs_off_fn = spec.get_pca_filename('Abs',name='WV Offline')
wv_abs_off = spec.load_spect_params(wv_abs_off_fn,nu=nu,wavelen=profs['WV Offline'].wavelength)

# O2-DIAL
o2_rb_on_fn = spec.get_pca_filename('RB',name='O2 Online')
o2_rb_on = spec.load_spect_params(o2_rb_on_fn,nu=nu,normalize=True)
o2_abs_on_fn = spec.get_pca_filename('Abs',name='O2 Online')
o2_abs_on = spec.load_spect_params(o2_abs_on_fn,nu=nu,wavelen=profs['O2 Online'].wavelength)

o2_rb_off_fn = spec.get_pca_filename('RB',name='O2 Offline')
o2_rb_off = spec.load_spect_params(o2_rb_off_fn,nu=nu,normalize=True)
o2_abs_off_fn = spec.get_pca_filename('Abs',name='O2 Offline')
o2_abs_off = spec.load_spect_params(o2_abs_off_fn,nu=nu,wavelen=profs['O2 Offline'].wavelength)

# build spectrscopy dictionaries
rb_spec = dict(zip(load_profile_list[0:2]+['HSRL']+load_profile_list[4:6],[wv_rb_off,wv_rb_on,hsrl_rb,o2_rb_on,o2_rb_off]))
abs_spec = dict(zip(load_profile_list[0:2]+load_profile_list[4:6],[wv_abs_off,wv_abs_on,o2_abs_on,o2_abs_off]))

# define etalon objects for each channel
Etalon_angle = 0.00*np.pi/180
WV_Filter = FO.FP_Etalon(1.0932e9,43.5e9,lp.c/wl['WV Online'],efficiency=0.95,InWavelength=False)

HSRL_Etalon_angle = 0.00*np.pi/180
HSRL_Filter1 = FO.FP_Etalon(5.7e9,45e9,lp.c/wl['HSRL Molecular'],efficiency=0.95,InWavelength=False)
HSRL_Filter2 = FO.FP_Etalon(15e9,250e9,lp.c/wl['HSRL Molecular'],efficiency=0.95,InWavelength=False)
HSRL_RbFilter = spec.RubidiumCellTransmission(nu+lp.c/wl['HSRL Molecular'],50+274.15,0,7.2e-2,iso87=0.97)

O2_Etalon_angle = 0.00*np.pi/180
O2_Filter = FO.FP_Etalon(1.0932e9,43.5e9,lp.c/wl['O2 Online'],efficiency=0.95,InWavelength=False)


# define receiver transmission function dictionary
Trx = {}
Trx['WV Online'] = WV_Filter.spectrum(nu+lp.c/wl['WV Online'],InWavelength=False)
Trx['WV Offline'] = WV_Filter.spectrum(nu+lp.c/wl['WV Offline'],InWavelength=False)
Trx['HSRL Mol'] = HSRL_Filter1.spectrum(nu+lp.c/wl['HSRL Molecular'],InWavelength=False) \
    *HSRL_Filter2.spectrum(nu+lp.c/wl['HSRL Molecular'],InWavelength=False) \
    *HSRL_RbFilter
Trx['HSRL Comb'] = HSRL_Filter1.spectrum(nu+lp.c/wl['HSRL Molecular'],InWavelength=False) \
    *HSRL_Filter2.spectrum(nu+lp.c/wl['HSRL Molecular'],InWavelength=False)
Trx['O2 Online'] = O2_Filter.spectrum(nu+lp.c/wl['O2 Online'],InWavelength=False)
Trx['O2 Offline'] = O2_Filter.spectrum(nu+lp.c/wl['O2 Offline'],InWavelength=False)

# define ln of BSR multipliers 
multBSR = {'WV':np.log((wl['WV Online']/wl['HSRL Molecular'])**3), 'O2':np.log((wl['O2 Online']/wl['HSRL Molecular'])**3)}



"""
Direct calculations of water vapor and BSR
"""
lam_on = np.ones(profs['WV Online'].time.size)*profs['WV Online'].wavelength # wavemeter bias -0.15e-12
lam_off = np.ones(profs['WV Offline'].time.size)*profs['WV Offline'].wavelength
nWV = wv.WaterVapor_2D(profs['WV Online'],profs['WV Offline'],lam_on,lam_off,pres,temp,error_order = 1)
#nWV = wv.WaterVapor_Simple(profs['WV Online'],profs['WV Offline'],pres.profile.flatten(),temp.profile.flatten())
#nWV.conv(0.0,np.sqrt(150**2+75**2)/nWV.mean_dR)
nWV.gain_scale(lp.N_A/lp.mH2O)
nWV.profile_type = '$m^{-3}$'

# poisson variable based processing
temp_i = np.interp(nWV.range_array,temp.range_array,temp.profile[0,:])
pres_i = np.interp(nWV.range_array,pres.range_array,pres.profile[0,:])

# load the PCA spectroscopy files and data          
sig_on_fn = spec.get_pca_filename('Abs',name='WV Online')
sig_on_data = spec.load_spect_params(sig_on_fn,wavelen=lam_on)
sig_on = spec.calc_pca_spectrum(sig_on_data,temp_i,pres_i)

sig_off_fn = spec.get_pca_filename('Abs',name='WV Offline')
sig_off_data = spec.load_spect_params(sig_off_fn,wavelen=lam_off)
sig_off = spec.calc_pca_spectrum(sig_off_data,temp_i,pres_i)

dsig = sig_on-sig_off
nWV_p = (-1.0/(2*(dsig)))*np.diff(rv.log_array(p_profs['WV Online'])-rv.log_array(p_profs['WV Offline']),axis=1)*(1.0/profs['WV Online'].mean_dR)
nWV_p = nWV_p



[eta_m,eta_c] = lp.RB_Efficiency([Trx['HSRL Mol'],Trx['HSRL Comb']],temp.profile.flatten(),pres.profile.flatten(),profs['HSRL Molecular'].wavelength,nu=nu,norm=True,max_size=10000)       
eta_m = eta_m.reshape(temp.profile.shape)  
#profs['HSRL Molecular'].gain_scale(MolGain,gain_var = (MolGain*0.05)**2)    
eta_c = eta_c.reshape(temp.profile.shape)
aer_beta,BSR,param_profs=wv.AerosolBackscatter(profs['HSRL Molecular'],profs['HSRL Combined'],beta_mol_sonde,negfilter=True,eta_am=Trx['HSRL Mol'][inu0],eta_ac=Trx['HSRL Comb'][inu0],eta_mm=eta_m,eta_mc=eta_c,gm=MolGain)

mol_p = -1*(p_profs['HSRL Molecular']*Trx['HSRL Comb'][inu0]-MolGain*p_profs['HSRL Combined']*Trx['HSRL Mol'][inu0])*(1.0/(MolGain*(Trx['HSRL Mol'][inu0]*eta_c-Trx['HSRL Comb'][inu0]*eta_m)))
aer_p = (p_profs['HSRL Molecular']*eta_c - MolGain*p_profs['HSRL Combined']*eta_m)*(1.0/(MolGain*(Trx['HSRL Mol'][inu0]*eta_c-Trx['HSRL Comb'][inu0]*eta_m)))
#com_p = (-1*Trx['HSRL Comb'][inu0]/(MolGain*(Trx['HSRL Mol'][inu0]*eta_c-Trx['HSRL Comb'][inu0]*eta_m))+eta_c/(MolGain*(Trx['HSRL Mol'][inu0]*eta_c-Trx['HSRL Comb'][inu0]*eta_m)))*p_profs['HSRL Molecular'] \
#        +(Trx['HSRL Mol'][inu0]*MolGain/(MolGain*(Trx['HSRL Mol'][inu0]*eta_c-Trx['HSRL Comb'][inu0]*eta_m))-MolGain*eta_m/(MolGain*(Trx['HSRL Mol'][inu0]*eta_c-Trx['HSRL Comb'][inu0]*eta_m)))*p_profs['HSRL Combined']

#BSR_p = aer_p/mol_p+1
#BSR_p = p_profs['HSRL Combined']/(MolGain*p_profs['HSRL Molecular'])

b_coeff = [-Trx['HSRL Comb'][inu0]/(MolGain*(Trx['HSRL Mol'][inu0]*eta_c-Trx['HSRL Comb'][inu0]*eta_m)),MolGain*Trx['HSRL Mol'][inu0]/(MolGain*(Trx['HSRL Mol'][inu0]*eta_c-Trx['HSRL Comb'][inu0]*eta_m))]
a_coeff = [eta_c/(MolGain*(Trx['HSRL Mol'][inu0]*eta_c-Trx['HSRL Comb'][inu0]*eta_m)),-MolGain*eta_m/(MolGain*(Trx['HSRL Mol'][inu0]*eta_c-Trx['HSRL Comb'][inu0]*eta_m))]
h = lambda x,y,nx,ny: rv.h_mix_ratio(x,y,nx,ny,a=a_coeff,b=b_coeff)
BSR_p = rv.h_two_var(p_profs['HSRL Molecular'],p_profs['HSRL Combined'],h)+1




## interpolate selected data onto retrieval grid
#interp_list = ['sim_T','sim_P','sim_nWV','BSR_HSRL_Molecular','Taer_HSRL_Molecular','Overlap','sim_beta_mol']
#interp_vars = {}
#for var in interp_list:
#    interp_vars[var] = np.interp(fit_profs['BSR'].range_array,sim_vars['sim_range'],sim_vars[var])
#    
##TAct = np.interp(fit_profs['BSR'].range_array,sim_vars['sim_range'],sim_vars['sim_T'])
#nWVAct = np.interp(fit_profs['BSR'].range_array,sim_vars['sim_range'],sim_vars['sim_nWV'])
BSRAct = np.interp(BSR.range_array,sim_vars['sim_range'],sim_vars['BSR_HSRL_Molecular'])



plt.figure(figsize=(6.4,8.0)); 
plt.fill_betweenx(nWV.range_array,
                  nWV.profile[0,:]-1*np.sqrt(nWV.profile_variance[0,:]), \
                  nWV.profile[0,:]+1*np.sqrt(nWV.profile_variance[0,:]), \
                  color='r',alpha=0.2,label='First Order')
plt.fill_betweenx(nWV.range_array, \
                  rv.moment(nWV_p[0,:],num=1)-1*np.sqrt(rv.moment(nWV_p[0,:],num=2)), \
                  rv.moment(nWV_p[0,:],num=1)+1*np.sqrt(rv.moment(nWV_p[0,:],num=2)),color='b',alpha=0.2,label='Third Order')
plt.plot(nWV.profile.flatten(),nWV.range_array,'b-')
plt.plot(rv.moment(nWV_p[0,:],num=0),nWV.range_array,'r-')
plt.plot(sim_vars['sim_nWV'],sim_vars['sim_range'],'k--',label='actual')
plt.xlabel('Water Vapor number density [$m^{-3}$]')
plt.ylabel('Altitude [m]')
plt.grid(b=True)
plt.xlim([0,4e24])

plt.figure(figsize=(6.4,8.0)); 
plt.semilogx(nWV.profile.flatten(),nWV.range_array,'b-')
plt.semilogx(np.sqrt(nWV.profile_variance.flatten()),nWV.range_array,'b:')
plt.semilogx(rv.moment(nWV_p[0,:],num=0),nWV.range_array,'r-')
plt.semilogx(np.sqrt(rv.moment(nWV_p[0,:],num=2)),nWV.range_array,'r:')
#plt.plot(sim_vars['sim_nWV'],sim_vars['sim_range'],'k--',label='actual')
plt.xlabel('Water Vapor number density [$m^{-3}$]')
plt.ylabel('Altitude [m]')
plt.grid(b=True)


plt.figure(figsize=(6.4,8.0)); 
plt.semilogx(BSR.profile.flatten(),BSR.range_array,'b-')
plt.semilogx(np.sqrt(BSR.profile_variance.flatten()),BSR.range_array,'b:')
plt.semilogx(rv.moment(BSR_p[0,:],num=0),BSR.range_array,'r-')
plt.semilogx(np.sqrt(rv.moment(BSR_p[0,:],num=2)),BSR.range_array,'r:')
plt.semilogx(BSRAct.flatten(),BSR.range_array,'k-',label='Actual')
#plt.plot(sim_vars['sim_nWV'],sim_vars['sim_range'],'k--',label='actual')
plt.xlabel('Backscatter Ratio')
plt.ylabel('Altitude [m]')
plt.grid(b=True)

plt.figure(figsize=(6.4,8.0)); 
plt.semilogx(np.ones(BSR_p.size).flatten(),BSR.range_array,'k--')
plt.semilogx(2*np.ones(BSR_p.size).flatten(),BSR.range_array,'k:')
plt.semilogx(((BSR.profile-1)/(np.sqrt(BSR.profile_variance))).flatten(),BSR.range_array,'b-',label='1st Order'); 
plt.semilogx((rv.moment(BSR_p[0,:],num=1)-1)/(np.sqrt(rv.moment(BSR_p[0,:],num=2))),BSR.range_array,'r--',label='3rd Order');
plt.xlabel('Backscatter Ratio SNR')
plt.ylabel('Altitude [m]')
plt.grid(b=True)

plt.figure(); 
plt.plot((BSR.profile.flatten()-BSRAct)/np.sqrt(BSR.profile_variance.flatten()),BSR.range_array,'b'); 
plt.plot((rv.moment(BSR_p[0,:],num=1)-BSRAct)/np.sqrt(rv.moment(BSR_p[0,:],num=2)),BSR.range_array,'r--');
plt.xlabel('Std. Normalized BSR Error')
plt.ylabel('Altitude [m]')
plt.grid(b=True)


plt.figure()
plt.plot(sim_vars['sim_nWV'],sim_vars['sim_range'],'k--',label='actual')
plt.plot(nWV.profile.flatten(),nWV.range_array,color='r',label='simulated')
plt.xlabel('Water Vapor number density [$m^{-3}$]')
plt.ylabel('Altitude [m]')
plt.grid(b=True)

plt.figure()
plt.plot(1e-6*sim_vars['sim_nu'],10*np.log10(sim_vars['WV_Online_laser_spectrum']/sim_vars['WV_Online_laser_spectrum'].max()))
plt.ylim([-60,0])
plt.grid(b=True)
plt.xlabel('Frequency [MHz]')
plt.title('Laser Spectrum')