# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 12:54:50 2018

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

import TDRetrieval2Lib as td

import datetime

import netCDF4 as nc4


ncfile = 'simulated_thermodynamic_DIAL_20180328_T1412_data.nc'

load_profile_list = ['WV Offline','WV Online','HSRL Combined','HSRL Molecular','O2 Online','O2 Offline']
load_var_list = ['T_WS','P_WS']

profs = {}
for prof_name in load_profile_list:
    profs[prof_name] = lp.load_nc_Profile(ncfile,('Simulated ' + prof_name).replace(' ','_'))
    
data_vars = lp.load_nc_vars(ncfile,load_var_list)

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

### raw signal convolution
##signal_list[0].conv(0.0,1.0/bin_res)
##signal_list[1].conv(0.0,1.0/bin_res)
#
#
#nWV = wv.WaterVapor_Simple(profs['WV Online'],profs['WV Offline'],pres.profile.flatten(),temp.profile.flatten())
#nWV.conv(0.0,np.sqrt(150**2+75**2)/nWV.mean_dR)




# etalon design
Etalon_angle = 0.00*np.pi/180
WV_Filter = FO.FP_Etalon(1.0932e9,43.5e9,lp.c/profs['WV Offline'].wavelength,efficiency=0.95,InWavelength=False)

HSRL_Etalon_angle = 0.00*np.pi/180
HSRL_Filter1 = FO.FP_Etalon(5.7e9,45e9,lp.c/profs['HSRL Molecular'].wavelength,efficiency=0.95,InWavelength=False)
HSRL_Filter2 = FO.FP_Etalon(15e9,250e9,lp.c/profs['HSRL Molecular'].wavelength,efficiency=0.95,InWavelength=False)

O2_Etalon_angle = 0.00*np.pi/180
O2_Filter = FO.FP_Etalon(1.0932e9,43.5e9,lp.c/profs['O2 Online'].wavelength,efficiency=0.95,InWavelength=False)


## Assume 97% Rb 87
#Lcell = 7.2e-2  # Rb Cell length in m
#Tcell = 50+274.15 # Rb Cell Temperature in K
#K85, K87 = spec.RubidiumD2Spectra(Tcell,nu+lp.c/profs['HSRL Molecular'].wavelength,0.0)  # Calculate absorption coefficeint
#
#
#RbFilter = np.exp(-Lcell*(np.sum(K87,axis=0)*0.97/0.27832+np.sum(K85,axis=0)*0.03/0.72172))
#TetalonHSRL = HSRL_Filter1.spectrum(nu+lp.c/profs['HSRL Molecular'].wavelength,InWavelength=False,aoi=0.0,transmit=True) \
#            *HSRL_Filter2.spectrum(nu+lp.c/profs['HSRL Molecular'].wavelength,InWavelength=False,aoi=0.0,transmit=True)           



#[eta_m,eta_c] = lp.RB_Efficiency([TetalonHSRL*RbFilter,TetalonHSRL],temp.profile.flatten(),pres.profile.flatten(),profs['HSRL Molecular'].wavelength,nu=nu,norm=True,max_size=10000)
#            
#eta_m = eta_m.reshape(temp.profile.shape)
#        
#profs['HSRL Molecular'].gain_scale(MolGain,gain_var = (MolGain*0.05)**2)
#    
#eta_c = eta_c.reshape(temp.profile.shape)
#
#aer_beta,BSR,param_profs=wv.AerosolBackscatter(profs['HSRL Molecular'],profs['HSRL Combined'],beta_mol_sonde,negfilter=True,eta_am=0.000,eta_ac=TetalonHSRL[inu0],eta_mm=eta_m,eta_mc=eta_c,gm=1.0)
#        
## use scalar for molecular gain
##molGainScale = 1.50  # scale for molecular channel
##signal_list[3].gain_scale(molGainScale)
#
## range dependent molecular gain adjustment
##plt.figure(); 
##plt.plot(signal_list[2].profile.flatten()/signal_list[3].profile.flatten())
##signal_list[3].diff_geo_overlap_correct(0.9855*3/7.0/eta_mol)
#mol = signal_list[3].copy()
#mol.gain_scale(1.69)
#comb = signal_list[2]
#aer_beta = lp.AerosolBackscatter(mol,comb,beta_mol_sonde)
#
#bsr = aer_beta.copy()
#bsr.profile = aer_beta.profile+beta_mol_sonde.profile
#bsr.divide_prof(beta_mol_sonde)
#bsr.label = 'Backscatter Ratio'
#bsr.descript = 'Backscatter Ratio at 780 nm'
#bsr.profile_type = ''
#
#
## direct calculation of o2 extinction coefficient
#alpha_o2 = nWV.copy()
#alpha_o2.profile = -1.0/(2)*np.diff(np.log(signal_list[4].profile/signal_list[5].profile),axis=1)/signal_list[5].mean_dR
#alpha_o2.label = 'Oxygen Differential Extinction'
#alpha_o2.descript = 'Oxygen Differential Extinction'
#alpha_o2.profile_type = '$m^{-1}$'
##nWV.range_array = range_diff
#alpha_o2.profile_variance = (0.5/signal_list[4].mean_dR)**2*( \
#    signal_list[4].profile_variance[:,1:]/signal_list[4].profile[:,1:]**2 + \
#    signal_list[4].profile_variance[:,:-1]/signal_list[4].profile[:,:-1]**2 + \
#    signal_list[5].profile_variance[:,1:]/signal_list[5].profile[:,1:]**2 + \
#    signal_list[5].profile_variance[:,:-1]/signal_list[5].profile[:,:-1]**2)
#
#
#
#plt.figure()
#plt.plot(sim_nWV/lp.N_A*lp.mH2O,sim_range*1e-3,'k--')
#plt.plot(nWV.profile.flatten(),nWV.range_array*1e-3,'r')
#plt.xlim([0,30])
#plt.xlabel('Absolute Humidity [$g/m^3$]')
#plt.ylabel('Altitude [km]')
#plt.grid(b=True)
#
#plt.figure()
#plt.plot(sim_T,sim_range*1e-3,'k--')
#plt.plot(temp.profile.flatten(),temp.range_array*1e-3,'b')
##plt.xlim([150,300])
#plt.xlabel('Temperature [$K$]')
#plt.ylabel('Altitude [km]')
#plt.grid(b=True)
#
#plt.figure()
#plt.semilogx(sim_beta_aer+sim_beta_cloud,sim_range*1e-3,'k--')
#plt.plot(aer_beta.profile.flatten(),aer_beta.range_array*1e-3,'r')
##plt.xlim([150,300])
#plt.xlabel('Particle Backscatter Coefficient [$m^{-1}sr^{-1}$]')
#plt.ylabel('Altitude [km]')
#plt.grid(b=True)
#
#plt.figure(); 
#plt.plot(alpha_o2.profile.flatten(),alpha_o2.range_array*1e-3);
#plt.xlim([0,6e-4])
#plt.xlabel('Oxygen Extinction [$m^{-1}$]')
#plt.ylabel('Altitude [km]')
#plt.grid(b=True)
#
#
#lp.LidarProfile(BS_Sig[np.newaxis,:],np.array([0]),label='Simulated '+name_list[ilist],lidar='WV-DIAL',binwidth=bin_res*2/lp.c,wavelength=wavelength_list[ilist],StartDate=datetime.datetime.today())
#
#range_limits = [200,6e3] # [500,6e3]
#
#nWVAct = nWV.copy()
#nWVAct.profile = np.interp(nWVAct.range_array,sim_range,sim_nWV)[np.newaxis,:]
#nWVAct.slice_range(range_lim=range_limits)
#nWVAct.label = 'Absolute Humidity'
#nWVAct.descript = 'Actual Absolute Humidity'
#nWVAct.profile_type = '$m^{-3}$'
#
#TAct = nWV.copy()
#TAct.profile = np.interp(TAct.range_array,sim_range,sim_T)[np.newaxis,:]
#TAct.slice_range(range_lim=range_limits)
#TAct.label = 'Temperature'
#TAct.descript = 'Actual Temperature'
#TAct.profile_type = 'K'
#
#BSRAct = nWV.copy()
#BSRAct.profile = np.interp(BSRAct.range_array,sim_range,BSR_780)[np.newaxis,:]
#BSRAct.slice_range(range_lim=range_limits)
#BSRAct.label = 'Backscatter Ratio'
#BSRAct.descript = 'Actual Backscatter Ratio at 780 nm'
#BSRAct.profile_type = ''
#
#
## compute signal ratios for optimization routines
#Hratio = signal_list[2].copy()
#Hratio.divide_prof(signal_list[3])
#Hratio.slice_range(range_lim=range_limits)
#
#Wratio = signal_list[1].copy()
#Wratio.divide_prof(signal_list[0])
#Wratio.slice_range(range_lim=range_limits)
#
#Oratio = signal_list[4].copy()
#Oratio.divide_prof(signal_list[5])
#Oratio.slice_range(range_lim=range_limits)
#
#presfit = pres.copy()
##presfit.gain_scale(101325)  
##presfit.profile_type = '$Pa$'
#presfit.slice_range(range_lim=range_limits)
#
#tempfit = temp.copy()
#tempfit.slice_range(range_lim=range_limits)
#
#nWVfit = nWV.copy()
#nWVfit.gain_scale(lp.N_A/lp.mH2O)
#nWVfit.slice_range(range_lim=range_limits)
#
#BSRfit = bsr.copy()
#BSRfit.slice_range(range_lim=range_limits)
##
##x02D = np.zeros((Oratio.range_array.size,3))
##x02D[:,0] = np.interp(Oratio.range_array,temp.range_array,temp.profile.flatten())
##x02D[:,1] = np.interp(Wratio.range_array,nWV.range_array,nWV.profile.flatten()*lp.N_A/lp.mH2O)
##x02D[:,2] = np.interp(Hratio.range_array,bsr.range_array,bsr.profile.flatten())
##
##ilim = x02D.shape[0]
##x0 = x02D.T.flatten()
##
##bnds = np.zeros((x0.size,2))
##bnds[:2*ilim,0] = 0
##bnds[2*ilim:,0] = 1.0
##bnds[:ilim,1] = 400.0
##bnds[ilim:2*ilim,1] = 1e25
##bnds[2*ilim:,1] = 1e5
##
##Herror = lambda x: (td.HSRLProfileRatio(x[:ilim],presfit.profile.flatten(),x[2*ilim:],Hratio.wavelength,HSRL_Filter1,GainRatio=0.9855*3/7.0)-Hratio.profile.flatten())**2/Hratio.profile_variance.flatten()
##Werror = lambda x: (td.WaterVaporProfileRatio(x[:ilim],presfit.profile.flatten(),x[ilim:2*ilim],x[2*ilim:],signal_list[1].wavelength,signal_list[0].wavelength,Filter,Wratio.mean_dR,GainRatio=1.0) \
##    -Wratio.profile.flatten())**2/Wratio.profile_variance.flatten()
##Oerror = lambda x: (td.OxygenProfileRatio(x[:ilim],presfit.profile.flatten(),x[ilim:2*ilim],x[2*ilim:],signal_list[4].wavelength,signal_list[5].wavelength,O2_Filter,Oratio.mean_dR,GainRatio=1.0) \
##    -Oratio.profile.flatten())**2/Oratio.profile_variance.flatten()
##    
##ProfError = lambda x: np.nansum(Herror(x)+Werror(x)+Oerror(x))
##
##
##
##
###x0 = np.zeros((Oratio.range_array.size+1,3))
###x02D = np.zeros((Oratio.range_array.size,3))
###x02D[1:,0] = np.interp(Oratio.range_array,temp.range_array,temp.profile.flatten())
###x02D[1:,1] = np.interp(Wratio.range_array,nWV.range_array,nWV.profile.flatten()*lp.N_A/lp.mH2O)
###x02D[1:,2] = np.interp(Hratio.range_array,bsr.range_array,bsr.profile.flatten())
###
###ilim = x02D.shape[0]
###x0 = x02D.T.flatten()
###
###bnds = np.zeros((x0.size,2))
###bnds[:2*ilim,0] = 0
###bnds[2*ilim:,0] = 1.0
###bnds[:ilim,1] = 400.0
###bnds[ilim:2*ilim,1] = 1e25
###bnds[2*ilim:,1] = 1e5
###bnds[0,0] = 0
###bnds[0,1] = 10
###bnds[ilim,0] = 0
###bnds[ilim,1] = 10
###bnds[2*ilim,0] = 0
###bnds[2*ilim,1] = 10
###
###Herror = lambda x: (td.HSRLProfileRatio(x[1:ilim],presfit.profile.flatten(),x[2*ilim+1:],Hratio.wavelength,HSRL_Filter1,GainRatio=x[2*ilim])-Hratio.profile.flatten())**2/Hratio.profile_variance.flatten()
###Werror = lambda x: (td.WaterVaporProfileRatio(x[1:ilim],presfit.profile.flatten(),x[ilim+1:2*ilim],x[2*ilim+1:],signal_list[1].wavelength,signal_list[0].wavelength,Filter,Wratio.mean_dR,GainRatio=x[ilim]) \
###    -Wratio.profile.flatten())**2/Wratio.profile_variance.flatten()
###Oerror = lambda x: (td.OxygenProfileRatio(x[1:ilim],presfit.profile.flatten(),x[ilim+1:2*ilim],x[2*ilim+1:],signal_list[4].wavelength,signal_list[5].wavelength,O2_Filter,Oratio.mean_dR,GainRatio=x[0]) \
###    -Oratio.profile.flatten())**2/Oratio.profile_variance.flatten()
###    
###ProfError = lambda x: Herror(x)+Werror(x)+Oerror(x)
##
##sol1D,opt_iterations,opt_exit_mode = scipy.optimize.fmin_tnc(ProfError,x0,bounds=bnds) #,maxfun=2000,eta=1e-5,disp=0)  
###sol2D = np.reshape(sol1D,x02D.shape)
#
#
"""
Analysis definitions
"""
#a_wavelength_list = [wavelength_list[0],wavelength_list[1],wavelength_list[2],wavelength_list[4],wavelength_list[5]]
##a_wavelength_list = [828.3026e-9,828.203e-9,780.246119e-9,769.2365e-9+0.001e-9,769.319768e-9]  # offline,online,Comb,Mol,online,offline  # O2 On center: 769.2339e-9
#a_name_list = ['WV Offline','WV Online','HSRL','O2 Online','O2 Offline']  # name corresponding to each wavelength

a_name_list = []
a_wavelength_list = []
for prof_name in profs.keys():
    a_wavelength_list.extend([profs[prof_name].wavelength])
    a_name_list.extend([str(prof_name)])


a_dnu = 20e6  # spacing in optical frequency space of simulation
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
wv_abs_on = spec.load_spect_params(wv_abs_on_fn,nu=nu)

wv_rb_off_fn = spec.get_pca_filename('RB',name='WV Offline')
wv_rb_off = spec.load_spect_params(wv_rb_off_fn,nu=nu,normalize=True)
wv_abs_off_fn = spec.get_pca_filename('Abs',name='WV Offline')
wv_abs_off = spec.load_spect_params(wv_abs_off_fn,nu=nu)

# O2-DIAL
o2_rb_on_fn = spec.get_pca_filename('RB',name='O2 Online')
o2_rb_on = spec.load_spect_params(o2_rb_on_fn,nu=nu,normalize=True)
o2_abs_on_fn = spec.get_pca_filename('Abs',name='O2 Online')
o2_abs_on = spec.load_spect_params(o2_abs_on_fn,nu=nu)

o2_rb_off_fn = spec.get_pca_filename('RB',name='O2 Offline')
o2_rb_off = spec.load_spect_params(o2_rb_off_fn,nu=nu,normalize=True)
o2_abs_off_fn = spec.get_pca_filename('Abs',name='O2 Offline')
o2_abs_off = spec.load_spect_params(o2_abs_off_fn,nu=nu)

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

# define BSR multipliers
multBSR = {'WV':(wl['WV Online']/wl['HSRL Molecular'])**3, 'O2':(wl['O2 Online']/wl['HSRL Molecular'])**3}



"""
Direct calculations of water vapor and BSR
"""
nWV = wv.WaterVapor_Simple(profs['WV Online'],profs['WV Offline'],pres.profile.flatten(),temp.profile.flatten())
nWV.conv(0.0,np.sqrt(150**2+75**2)/nWV.mean_dR)
nWV.gain_scale(lp.N_A/lp.mH2O)
nWV.profile_type = '$m^{-3}$'

[eta_m,eta_c] = lp.RB_Efficiency([Trx['HSRL Mol'],Trx['HSRL Comb']],temp.profile.flatten(),pres.profile.flatten(),profs['HSRL Molecular'].wavelength,nu=nu,norm=True,max_size=10000)       
eta_m = eta_m.reshape(temp.profile.shape)  
#profs['HSRL Molecular'].gain_scale(MolGain,gain_var = (MolGain*0.05)**2)    
eta_c = eta_c.reshape(temp.profile.shape)
aer_beta,BSR,param_profs=wv.AerosolBackscatter(profs['HSRL Molecular'],profs['HSRL Combined'],beta_mol_sonde,negfilter=True,eta_am=Trx['HSRL Mol'][inu0],eta_ac=Trx['HSRL Comb'][inu0],eta_mm=eta_m,eta_mc=eta_c,gm=MolGain)




"""
Optimization Routines
"""

range_limits = [200,6e3] # [500,6e3]

fit_profs = {}
for var in profs.keys():
    fit_profs[var] = profs[var].copy()
fit_profs['nWV'] = nWV.copy()
fit_profs['BSR'] = BSR.copy()
fit_profs['P'] = pres.copy()
fit_profs['T'] = temp.copy()



fit_data = {}
for var in fit_profs.keys():
    fit_profs[var].slice_range(range_lim=range_limits)

# interpolate water vapor data onto the same range grid as the other data points
nWVi = np.interp(fit_profs['BSR'].range_array,nWV.range_array,nWV.profile.flatten())

# copy arrays of data to be fit from lidar profiles (including background)
fit_data['HSRL Mol'] = fit_profs['HSRL Molecular'].profile.flatten()+fit_profs['HSRL Molecular'].bg
fit_data['HSRL Comb'] = fit_profs['HSRL Combined'].profile.flatten()+fit_profs['HSRL Combined'].bg
fit_data['WV Online'] = fit_profs['WV Online'].profile.flatten()+fit_profs['WV Online'].bg
fit_data['WV Offline'] = fit_profs['WV Offline'].profile.flatten()+fit_profs['WV Offline'].bg
fit_data['O2 Online'] = fit_profs['O2 Online'].profile.flatten()+fit_profs['O2 Online'].bg
fit_data['O2 Offline'] = fit_profs['O2 Offline'].profile.flatten()+fit_profs['O2 Offline'].bg

fit_data['HSRL Mol BG'] = fit_profs['HSRL Molecular'].bg
fit_data['HSRL Comb BG'] = fit_profs['HSRL Combined'].bg
fit_data['WV Online BG'] = fit_profs['WV Online'].bg
fit_data['WV Offline BG'] = fit_profs['WV Offline'].bg
fit_data['O2 Online BG'] = fit_profs['O2 Online'].bg
fit_data['O2 Offline BG'] = fit_profs['O2 Offline'].bg

# compute signal ratios for optimization routines
#Hratio = signal_list[2].copy()
#Hratio.divide_prof(signal_list[3])
#Hratio.slice_range(range_lim=range_limits)

#Prof = {}
#ProfVar = {}
#Prof['HSRL'] = Hratio.profile.flatten()
#ProfVar['HSRL'] = Hratio.profile_variance.flatten()
#
#Prof['WV'] = Wratio.profile.flatten()
#ProfVar['WV'] = Wratio.profile_variance.flatten()
#
#Prof['O2'] = Oratio.profile.flatten()
#ProfVar['O2'] = Oratio.profile_variance.flatten()
#
# index into laser frequency
inuL = {}
inuL['HSRL'] = inu0
inuL['WV Online'] = inu0
inuL['WV Offline'] = inu0
inuL['O2 Online'] = inu0
inuL['O2 Offline'] = inu0

#xAct = np.zeros((Hratio.range_array.size+1,3))
#xAct[0,0] = K_list[2]/K_list[3]
#xAct[0,1] = 1.0
#xAct[0,2] = 1.0
#xAct[1:,0] = TAct.profile.flatten()
#xAct[1:,1] = nWVAct.profile.flatten()
#xAct[1:,2] = BSRAct.profile.flatten()

x0 = np.zeros((fit_data['HSRL Mol'].size+1,6))
x0[0,0] = 2.0 # HSRL Molecular
x0[0,1] = 1.0 # HSRL Combined
x0[0,2] = 1.0 # WV Online
x0[0,3] = 1.0 # WV Offline
x0[0,4] = 1.0 # O2 Online
x0[0,5] = 1.0 # O2 Offline
x0[1:,0] = fit_profs['T'].profile.flatten()
x0[1:,1] = nWVi.flatten()  # use water vapor data on interpolated grid to align with other data points
x0[1:,2] = fit_profs['BSR'].profile.flatten()
# currently I am not initializing the phi variables to anything other than 0

bndL = -500*np.ones(x0.shape)
bndU = 500*np.ones(x0.shape)
#bndL[0,0] = x0[0,0]*0.8
#bndU[0,0] = x0[0,0]*1.2
#bndL[0,1] = x0[0,1]*0.8
#bndU[0,1] = x0[0,1]*1.2
#bndL[0,2] = x0[0,2]*0.8
#bndU[0,2] = x0[0,1]*1.2

bndL[1:,0] = 180
bndU[1:,0] = 310
bndL[1:,1] = 0
bndU[1:,1] = 10e24
bndL[1:,2] = 1
bndU[1:,2] = 1e5

bnds = np.hstack((bndL.flatten()[:,np.newaxis],bndU.flatten()[:,np.newaxis]))

x0i = x0.flatten()  # initial guess of x without phi.  This is to come up with an estimate for phi

# estimate profiles with phi = 0 
fit_test = td.TDProfiles(fit_data,x0i,Trx,rb_spec,abs_spec,fit_profs['WV Online'].mean_dR,inuL,multBSR,temp.profile[0,0],pres.profile[0,0])

# estimate phi for each profile set
phiH = 0.5*(np.log(fit_data['HSRL Mol']/fit_test[0])+np.log(fit_data['HSRL Comb']/fit_test[1]))
plt.figure()
plt.plot(np.log(fit_data['HSRL Mol']/fit_test[0]))
plt.plot(np.log(fit_data['HSRL Comb']/fit_test[1]))
plt.plot(phiH,'--')

phiW = 0.5*(np.log(fit_data['WV Online']/fit_test[2])+np.log(fit_data['WV Offline']/fit_test[3]))
plt.figure()
plt.plot(np.log(fit_data['WV Online']/fit_test[2]))
plt.plot(np.log(fit_data['WV Offline']/fit_test[3]))
plt.plot(phiW,'--')

phiO = 0.5*(np.log(fit_data['O2 Online']/fit_test[4])+np.log(fit_data['O2 Offline']/fit_test[5]))
plt.figure()
plt.plot(np.log(fit_data['O2 Online']/fit_test[4]))
plt.plot(np.log(fit_data['O2 Offline']/fit_test[5]))
plt.plot(phiO,'--')

# fill in estimate of phi for initial guess x0
x0[1:,3] = phiH
x0[1:,4] = phiW
x0[1:,5] = phiO

x0 = x0.flatten()

## set out of bounds values to bound limits
#iL = np.nonzero(x0<bndL.flatten())[0]
#x0[iL] = bnds[iL,0]
#iU = np.nonzero(x0<bndU.flatten())[0]
#x0[iU] = bnds[iU,1]

#Efit = lambda x: td.TDErrorFunction(Prof,ProfVar,x,presfit.profile.flatten(),Trx,rb_spec,abs_spec,Hratio.mean_dR,inuL,multBSR)
#gradE = lambda x: td.TDGradientFunction(Prof,ProfVar,x0,presfit.profile.flatten(),Trx,rb_spec,abs_spec,Hratio.mean_dR,inuL,multBSR)

Efit = lambda x: td.TDErrorFunction(fit_data,x,Trx,rb_spec,abs_spec,fit_profs['WV Online'].mean_dR,inuL,multBSR,temp.profile[0,0],pres.profile[0,0])
gradE = lambda x: td.TDGradientFunction(fit_data,x,Trx,rb_spec,abs_spec,fit_profs['WV Online'].mean_dR,inuL,multBSR,temp.profile[0,0],pres.profile[0,0])
gradEnum = td.Num_Gradient(Efit,x0,step_size=1e-3)

# check to make sure gradient function is consistent with numerically calculated gradient
plt.figure(); 
plt.plot(gradE(x0),label='analytic'); 
plt.plot(gradEnum,'--',label='numeric')
plt.title('Gradient Compare')
plt.legend()

# TDProfiles(Prof,x,Trx,rb_spec,abs_spec,dr,inu0,bsrMult,base_T,base_P)

# HSRL_mol, HSRL_comb, WV_on, WV_off, O2_on, O2_off
#plt.figure()
#plt.semilogy(fit_test[0])
#plt.semilogy(fit_data['HSRL Mol'])
#
#plt.figure()
#plt.semilogy(fit_test[1])
#plt.semilogy(fit_data['HSRL Comb'])




sol1D,opt_iterations,opt_exit_mode = scipy.optimize.fmin_tnc(Efit,x0,fprime=gradE) #,bounds=bnds) #,maxfun=2000,eta=1e-5,disp=0)  
print(scipy.optimize.tnc.RCSTRINGS[opt_exit_mode])

sol2D = sol1D.reshape((fit_data['HSRL Mol'].size+1,6))  # reshape solution into an easier to interpret 2D array

# Build profiles based on the solution
fit_eval = td.TDProfiles(fit_data,sol1D,Trx,rb_spec,abs_spec,fit_profs['WV Online'].mean_dR,inuL,multBSR,temp.profile[0,0],pres.profile[0,0])
ch_names = ['HSRL Mol','HSRL Comb','WV Online','WV Offline','O2 Online','O2 Offline']

# plot the fits on top of the data
for ai,ch in enumerate(ch_names):
    plt.figure()
    plt.semilogy(fit_data[ch]-fit_data[ch+' BG'],label='obs')
    plt.semilogy(fit_eval[ai],label='fit')
    plt.title(ch)
    plt.legend()

plt.figure()
plt.plot(fit_profs['T'].profile.flatten(),fit_profs['T'].range_array,label='initial guess')
plt.plot(sol2D[1:,0],fit_profs['T'].range_array,label='estimate')
plt.xlabel('Temperature [K]')
plt.ylabel('Range [m]')
plt.legend()

plt.figure()
plt.plot(fit_profs['nWV'].profile.flatten(),fit_profs['nWV'].range_array,label='initial guess')
plt.plot(sol2D[1:,1],fit_profs['BSR'].range_array,label='estimate')
plt.xlabel('Water Vapor Density [$m^{3}$]')
plt.ylabel('Range [m]')
plt.legend()

plt.figure()
plt.semilogx(fit_profs['BSR'].profile.flatten(),fit_profs['BSR'].range_array,label='initial guess')
plt.semilogx(sol2D[1:,2],fit_profs['BSR'].range_array,label='estimate')
plt.xlabel('Backscatter Ratio')
plt.ylabel('Range [m]')
plt.legend()

#plt.figure()
#plt.semilogy(fit_data['HSRL Mol']-fit_data['HSRL Mol BG'])
#plt.semilogy(fit_eval[0])
#
#plt.figure()
#plt.semilogy(fit_data['HSRL Comb']-fit_data['HSRL Comb BG'])
#plt.semilogy(fit_eval[1])
#
#plt.figure()
#plt.semilogy(fit_data['WV Online']-fit_data['WV Online BG'])
#plt.semilogy(fit_eval[2])
#
#plt.figure()
#plt.semilogy(fit_data['WV Offline']-fit_data['WV Offline BG'])
#plt.semilogy(fit_eval[3])
#
#plt.figure()
#plt.semilogy(fit_data['O2 Online']-fit_data['O2 Online BG'])
#plt.semilogy(fit_eval[4])
#
#plt.figure()
#plt.semilogy(fit_data['O2 Offline']-fit_data['O2 Offline BG'])
#plt.semilogy(fit_eval[5])


## Check ratio functions:
#fit_test =td.TDRatios(Prof,xAct,presfit.profile.flatten(),Trx,rb_spec,abs_spec,Hratio.mean_dR,inuL,multBSR)
#plt.figure()
#plt.semilogx(fit_test[0],Hratio.range_array,label='Fit Estimate')
#plt.semilogx(Hratio.profile.flatten(),Hratio.range_array,label='Signal Ratio')
#plt.semilogx(BSRAct.profile.flatten(),BSRAct.range_array,'k--',label='Actual BSR')
#plt.title('HSRL Ratio')
#plt.grid(b=True)
#plt.legend()
#
#plt.figure()
#plt.semilogx(fit_test[1],Wratio.range_array,label='Fit Estimate')
#plt.semilogx(Wratio.profile.flatten(),Wratio.range_array,label='Signal Ratio')
#plt.title('Water Vapor Ratio')
#plt.grid(b=True)
#plt.legend()
#
#plt.figure()
#plt.semilogx(1./fit_test[2],Oratio.range_array,label='Fit Estimate')
#plt.semilogx(Oratio.profile.flatten(),Oratio.range_array,label='Signal Ratio')
#plt.title('Oxygen Ratio')
#plt.grid(b=True)
#plt.legend()
#
#"""
#
#sol1D,opt_iterations,opt_exit_mode = scipy.optimize.fmin_tnc(Efit,x0,fprime=gradE,bounds=bnds) #,bounds=bnds) #,maxfun=2000,eta=1e-5,disp=0)  
#sol2D = sol1D.reshape((Hratio.range_array.size+1,3))
#
#fits = td.TDRatios(Prof,sol1D,presfit.profile.flatten(),Trx,rb_spec,abs_spec,Hratio.mean_dR,inuL,multBSR)
#
#print(scipy.optimize.tnc.RCSTRINGS[opt_exit_mode])
#
#plt.figure()
#plt.plot(sim_nWV/lp.N_A*lp.mH2O,sim_range*1e-3,'k--')
#plt.plot(nWV.profile.flatten(),nWV.range_array*1e-3,'r')
#plt.plot(sol2D[1:,1]/lp.N_A*lp.mH2O,Hratio.range_array*1e-3,'g.')
#plt.xlim([0,30])
#plt.xlabel('Absolute Humidity [$g/m^3$]')
#plt.ylabel('Altitude [km]')
#plt.grid(b=True)
#
#plt.figure()
#plt.plot(sim_T,sim_range*1e-3,'k--')
#plt.plot(temp.profile.flatten(),temp.range_array*1e-3,'b')
#plt.plot(sol2D[1:,0],Hratio.range_array*1e-3,'g.')
##plt.xlim([150,300])
#plt.xlabel('Temperature [$K$]')
#plt.ylabel('Altitude [km]')
#plt.grid(b=True)
#
#plt.figure()
#plt.semilogx((sim_beta_aer+sim_beta_cloud+sim_beta_mol)/sim_beta_mol,sim_range*1e-3,'k--')
#plt.semilogx(bsr.profile.flatten(),bsr.range_array*1e-3,'r')
#plt.semilogx(sol2D[1:,2],Hratio.range_array*1e-3,'g.')
##plt.xlim([150,300])
#plt.xlabel('Particle Backscatter Coefficient [$m^{-1}sr^{-1}$]')
#plt.ylabel('Altitude [km]')
#plt.grid(b=True)
#"""
#
#
#"""
#Compare assumed and actual profiles
#"""
#
#"""
#HSRL Validation Process
#Issue:  HSRL ratio was not agreeing with observation even when inserting the simulation
#    parameters.
#Resolution: molecular backscatter spectrum was not being normalized properly in the pca
#    spectroscopy library.  If the frequency grid was different than the grid it trained
#    on, the normalization needed to change.  There is now a normalize option in the 
#    function that should be True for molecular backscatter spectrum calculation
#"""
#
#HSRLmol = td.HSRLProfile(TAct.profile.flatten(),presfit.profile.flatten(),BSRAct.profile.flatten(),rb_spec['HSRL'],Trx['HSRL Mol'],inuL['HSRL'],1.0)
#
#HSRLcomb = td.HSRLProfile(TAct.profile.flatten(),presfit.profile.flatten(),BSRAct.profile.flatten(),rb_spec['HSRL'],Trx['HSRL Comb'],inuL['HSRL'],1.0)
#
#HSRL_Ratio = td.HSRLProfileRatio(TAct.profile.flatten(),presfit.profile.flatten(),BSRAct.profile.flatten(),Trx['HSRL Mol'],Trx['HSRL Comb'],rb_spec['HSRL'],inuL['HSRL'],GainRatio=1.0/2.1)
#
#TetalonHSRL = HSRL_Filter1.spectrum(sim_nu+lp.c/wavelength_list[2],InWavelength=False,aoi=Etalon_angle,transmit=True) \
#            *HSRL_Filter2.spectrum(sim_nu+lp.c/wavelength_list[2],InWavelength=False,aoi=Etalon_angle,transmit=True)
#
#beta_mol_780 = 5.45*(550.0e-9/wavelength_list[3])**4*1e-32*(sim_P/9.86923e-6)/(sim_T*lp.kB)
#beta_mol = 5.45*(550.0e-9/wavelength_list[ilist])**4*1e-32*(sim_P/9.86923e-6)/(sim_T*lp.kB)
#beta_aer = sim_beta_aer*wavelength_list[2]/780.24e-9
#beta_cloud = sim_beta_cloud  #*wavelength_list[ilist]/780.24e-9  # No wavelength dependence in cloud
#alpha_aer = beta_aer*sim_LR + beta_cloud*sim_cloud_LR
#OD_aer = np.cumsum(alpha_aer)*sim_dr
#OD_mol = np.cumsum(beta_mol*8*np.pi/3)*sim_dr
#Taer = np.exp(-OD_aer-OD_mol)
#
#plt.figure()
#plt.semilogx(beta_mol_780,sim_range,label='actual profile')
#plt.semilogx(beta_mol_sonde.profile.flatten(),beta_mol_sonde.range_array,label='assumed profile')
#plt.xlabel(r'$\beta_m(R)$ [$m^{-1}sr^{-1}$]')
#plt.grid(b=True)
#
#beta_mol_sub = np.interp(TAct.range_array,sim_range,beta_mol_780)
#Overlap_sub = np.interp(TAct.range_array,sim_range,Overlap)
#beta_aer_sub = np.interp(TAct.range_array,sim_range,beta_aer+beta_cloud)
#OD_aer_sub = np.interp(TAct.range_array,sim_range,OD_aer)
#OD_mol_sub = np.interp(TAct.range_array,sim_range,OD_mol)
#Taer_sub = np.exp(-OD_aer_sub-OD_mol_sub)
#
#molBeta_nu = lp.RB_Spectrum(sim_T,sim_P,wavelength_list[2],nu=sim_nu,norm=True)
#betaM_sub = spec.calc_pca_spectrum(rb_spec['HSRL'],TAct.profile.flatten(),presfit.profile.flatten())
#betaM_sub_norm = np.sum(betaM_sub,axis=0)
#
#i_plt = 10
#plot_alt = TAct.range_array[i_plt]
#isim = np.argmin(np.abs(sim_range-plot_alt))
#plt.figure()
#plt.plot(nu,betaM_sub[:,i_plt],'.')
#plt.plot(sim_nu,molBeta_nu[:,isim])
#plt.xlabel('Molecular Backscatter Spectrum')
#plt.title('%d m'%plot_alt)
#
#plt.figure()
#plt.plot(nu,Trx['HSRL Mol'],'.')
#plt.plot(sim_nu,TetalonHSRL*RbFilter)
#plt.xlabel('Molecular Channel Transmission Spectrum')
#
#
#T_rx_mol_sub = np.sum(Trx['HSRL Mol'][:,np.newaxis]*betaM_sub,axis=0)
#T_rx_mol = np.sum(molBeta_nu*TetalonHSRL[:,np.newaxis]*RbFilter[:,np.newaxis],axis=0)
#
#plt.figure();
#plt.plot(nu,(Trx['HSRL Mol'][:,np.newaxis]*betaM_sub)[:,i_plt])
#plt.plot(sim_nu,(molBeta_nu*TetalonHSRL[:,np.newaxis]*RbFilter[:,np.newaxis])[:,isim])
#
#plt.figure()
#plt.plot(T_rx_mol_sub,TAct.range_array)
#plt.plot(T_rx_mol,sim_range)
#plt.xlabel('Molecular Transmission Efficiency')
#
#plt.figure()
#plt.semilogx(K_list[3]*Taer_sub*Overlap_sub*beta_mol_sub*BSRAct.profile.flatten()*HSRLmol/BSRAct.range_array**2,BSRAct.range_array)
#plt.semilogx(signal_list[3].profile.flatten(),signal_list[3].range_array)
#
#plt.figure()
#plt.semilogx(K_list[2]*Taer_sub*Overlap_sub*beta_mol_sub*BSRAct.profile.flatten()*HSRLcomb/BSRAct.range_array**2,BSRAct.range_array)
#plt.semilogx(signal_list[2].profile.flatten(),signal_list[2].range_array)
#
#plt.figure()
#plt.plot((K_list[2]*Taer_sub*Overlap_sub*beta_mol_sub*BSRAct.profile.flatten()*HSRLcomb/BSRAct.range_array**2)/(K_list[3]*Taer_sub*Overlap_sub*beta_mol_sub*BSRAct.profile.flatten()*HSRLmol/BSRAct.range_array**2),BSRAct.range_array,label='Model from Profiles')
#plt.plot(Hratio.profile.flatten(),Hratio.range_array,label='Simulation')
#plt.plot(HSRL_Ratio,BSRAct.range_array,label='TD function')
#
#"""
#WV Validation Process
#Issue:  It looks like the optical depth of the water vapor signal is causing
#    a multiplier error between the observed signal and that obtained by the
#    model.
#Possible Resolution:  It looks like this multiplier is traced to the region
#    between the first valid range gate and the lidar, where optical depth
#    from this altitude region was not included in the calculations.  The
#    question is how to integrate that into the routine.  It may need to include
#    interpolation from the surface station to the first gate.
#"""
#
#molBeta_nu = lp.RB_Spectrum(sim_T,sim_P,wavelength_list[1],nu=sim_nu,norm=True)
#betaM_sub = spec.calc_pca_spectrum(rb_spec['WV Online'],TAct.profile.flatten(),presfit.profile.flatten())
#
#TetalonWVOn = Filter.spectrum(sim_nu+lp.c/wavelength_list[1],InWavelength=False,aoi=Etalon_angle,transmit=True)  
#
## obtain frequency resolved water vapor extinction coefficient
#ext_wv = lp.WV_ExtinctionFromHITRAN(lp.c/wavelength_list[1]+sim_nu,sim_T,sim_P,nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9]),freqnorm=True).T
#    
#OD_wv = np.cumsum(sim_nWV[np.newaxis,:]*ext_wv,axis=1)*sim_dr  # obtain frequency resolved optical depth
#
#T_tx = np.exp(-OD_wv[inu0,:])  # outgoing atmospheric transmission
#T_rx_aer = np.exp(-OD_wv[inu0,:])*Tetalon[inu0]  # return transmission for atmosphere and etalon seen by aerosols
#T_rx_mol = np.sum(molBeta_nu*TetalonWVOn[:,np.newaxis]*np.exp(-OD_wv),axis=0)  
#
#sigS = spec.calc_pca_spectrum(abs_spec['WV Online'],TAct.profile.flatten(),presfit.profile.flatten())
#r_int = np.arange(0,TAct.range_array[0],TAct.mean_dR)  # interpolated ranges between surface station and first range gate
#Tint = np.interp(r_int,np.array([0,TAct.range_array[0]]),np.array([sim_T[0],TAct.profile[0,0]]))
#Pint = np.interp(r_int,presfit.range_array,presfit.profile.flatten())
#nWVint = np.interp(r_int,np.array([0,TAct.range_array[0]]),np.array([sim_nWV[0],nWVAct.profile[0,0]]))
#ODs0 = np.sum(nWVint[np.newaxis,:]*spec.calc_pca_spectrum(abs_spec['WV Online'],Tint,Pint),axis=1)*TAct.mean_dR
##ODs0 = sim_nWV[0]*spec.calc_pca_spectrum(abs_spec['WV Online'],sim_T[0],sim_P[0])*nWVAct.range_array[0]
#ODs = np.cumsum(nWVAct.profile*sigS,axis=1)*TAct.mean_dR+ODs0[:,np.newaxis]
#Ts = np.exp(-ODs)
##np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0)
#
#i_plt = 0
#plot_alt = TAct.range_array[i_plt]
#isim = np.argmin(np.abs(sim_range-plot_alt))
#
#plt.figure()
#plt.plot(nu,betaM_sub[:,i_plt],'.')
#plt.plot(sim_nu,molBeta_nu[:,isim])
#plt.xlabel('Molecular Backscatter Spectrum')
#plt.title('%d m'%plot_alt)
#
#plt.figure()
#plt.plot(nu,Trx['WV Online'],'.')
#plt.plot(sim_nu,TetalonWVOn)
#plt.xlabel('Molecular Channel Transmission Spectrum')
#
#plt.figure(); 
#plt.plot(nu,(nWVAct.profile*sigS)[:,i_plt]); 
#plt.plot(sim_nu,(sim_nWV[np.newaxis,:]*ext_wv)[:,isim],'--')
#plt.title('WV Online Spectrum at %d m'%plot_alt)
#
#T_rx_wvon_sub = np.sum(Trx['WV Online'][:,np.newaxis]*betaM_sub,axis=0)
#T_rx_wvon = np.sum(molBeta_nu*TetalonWVOn[:,np.newaxis],axis=0)
#
#plt.figure()
#plt.plot(T_rx_wvon_sub,TAct.range_array)
#plt.plot(T_rx_wvon,sim_range,'--')
#plt.xlabel('Molecular Transmission Efficiency')