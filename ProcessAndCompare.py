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
import WVProfileFunctions as wv
import FourierOpticsLib as FO
import SpectrumLib as spec

import TDRetrieval3Lib as td # use the TD3 library that treats state variables as exponents (e.g. ln(T) = x)

import datetime

#import netCDF4 as nc4

o2_spec_file = '/Users/mhayman/Documents/DIAL/O2_HITRAN2012_760_781.txt'

#ncfile = 'simulated_thermodynamic_DIAL_20180331_T1256_sim.nc'
#ncfile = 'simulated_thermodynamic_DIAL_20180411_T1125_sim.nc'
#ncfile = 'simulated_thermodynamic_DIAL_20180411_T1144_sim.nc'
#ncfile = 'simulated_thermodynamic_DIAL_20180411_T1231_sim.nc'
#ncfile = 'simulated_thermodynamic_DIAL_20180411_T1254_sim.nc'
#ncfile = 'simulated_thermodynamic_DIAL_20180411_T1305_sim.nc'
#ncfile = 'simulated_thermodynamic_DIAL_20180411_T1328_sim.nc'


#ncfile = 'simulated_thermodynamic_DIAL_20180411_T1347_'  # 1 GHz HWHM Laser Spectrum, no shot noise
#ncfile = 'simulated_thermodynamic_DIAL_20180411_T1350_'  # 6 GHz HWHM Laser Spectrum, no shot noise
#ncfile = 'simulated_thermodynamic_DIAL_20180411_T1357_'  # 6 GHz HWHM Laser Spectrum, no shot noise 5% out of band blocking
#ncfile = 'simulated_thermodynamic_DIAL_20180411_T1358_'  # 1 GHz HWHM Laser Spectrum, no shot noise 5% out of band blocking
#ncfile = 'simulated_thermodynamic_DIAL_20180411_T1404_'  # 1 GHz HWHM Laser Spectrum, no shot noise 1% out of band blocking, 100 MHz etalon shift
#ncfile = 'simulated_thermodynamic_DIAL_20180411_T1407_'  # 1 GHz HWHM Laser Spectrum, no shot noise 1% out of band blocking, 100 MHz etalon shift
ncfile = 'simulated_thermodynamic_DIAL_20180412_T1344_'  # 1 GHz HWHM Laser Spectrum, with shot noise 1% out of band blocking, 100 MHz etalon shift


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
for var in profs.keys():
    raw_profs[var] = profs[var].copy()
    profs[var].bg_subtract(-50)

lp.plotprofiles(profs)

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
lam_on = np.ones(profs['WV Online'].time.size)*profs['WV Online'].wavelength
lam_off = np.ones(profs['WV Offline'].time.size)*profs['WV Offline'].wavelength
nWV = wv.WaterVapor_2D(profs['WV Online'],profs['WV Offline'],lam_on,lam_off,pres,temp,error_order = 1)
#nWV = wv.WaterVapor_Simple(profs['WV Online'],profs['WV Offline'],pres.profile.flatten(),temp.profile.flatten())
nWV.conv(0.0,np.sqrt(150**2+75**2)/nWV.mean_dR)
nWV.gain_scale(lp.N_A/lp.mH2O)
nWV.profile_type = '$m^{-3}$'

[eta_m,eta_c] = lp.RB_Efficiency([Trx['HSRL Mol'],Trx['HSRL Comb']],temp.profile.flatten(),pres.profile.flatten(),profs['HSRL Molecular'].wavelength,nu=nu,norm=True,max_size=10000)       
eta_m = eta_m.reshape(temp.profile.shape)  
#profs['HSRL Molecular'].gain_scale(MolGain,gain_var = (MolGain*0.05)**2)    
eta_c = eta_c.reshape(temp.profile.shape)
aer_beta,BSR,param_profs=wv.AerosolBackscatter(profs['HSRL Molecular'],profs['HSRL Combined'],beta_mol_sonde,negfilter=True,eta_am=Trx['HSRL Mol'][inu0],eta_ac=Trx['HSRL Comb'][inu0],eta_mm=eta_m,eta_mc=eta_c,gm=MolGain)


## interpolate selected data onto retrieval grid
#interp_list = ['sim_T','sim_P','sim_nWV','BSR_HSRL_Molecular','Taer_HSRL_Molecular','Overlap','sim_beta_mol']
#interp_vars = {}
#for var in interp_list:
#    interp_vars[var] = np.interp(fit_profs['BSR'].range_array,sim_vars['sim_range'],sim_vars[var])
#    
##TAct = np.interp(fit_profs['BSR'].range_array,sim_vars['sim_range'],sim_vars['sim_T'])
#nWVAct = np.interp(fit_profs['BSR'].range_array,sim_vars['sim_range'],sim_vars['sim_nWV'])
##BSRAct = np.interp(fit_profs['BSR'].range_array,sim_vars['sim_range'],sim_vars['BSR_HSRL_Molecular'])


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