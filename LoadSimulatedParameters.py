# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:07:08 2018

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

ncfile = 'simulated_thermodynamic_DIAL_20180331_T1256_sim.nc'

load_sim_list = ['sim_T','sim_P','sim_range','sim_nu','Tetalon_O2_Online', 
    'Tetalon_O2_Offline','sim_nWV','BSR_HSRL_Molecular','Taer_HSRL_Molecular',
    'Overlap','sim_beta_mol','RbFilter','Tetalon_HSRL_Molecular',
    'Tetalon_WV_Online','Tetalon_WV_Offline','sim_nO2','sim_dr',
    'T_tx_O2_Online','T_rx_aer_O2_Online','T_rx_mol_O2_Online']
    
sim_vars = lp.load_nc_vars(ncfile,load_sim_list)

# interpolate selected data onto retrieval grid
interp_list = ['sim_T','sim_P','sim_nWV','BSR_HSRL_Molecular','Taer_HSRL_Molecular','Overlap','sim_beta_mol']
interp_vars = {}
for var in interp_list:
    interp_vars[var] = np.interp(fit_profs['BSR'].range_array,sim_vars['sim_range'],sim_vars[var])
    
#TAct = np.interp(fit_profs['BSR'].range_array,sim_vars['sim_range'],sim_vars['sim_T'])
#nWVAct = np.interp(fit_profs['BSR'].range_array,sim_vars['sim_range'],sim_vars['sim_nWV'])
#BSRAct = np.interp(fit_profs['BSR'].range_array,sim_vars['sim_range'],sim_vars['BSR_HSRL_Molecular'])



"""
Testing fit functions
requires running ProcessTDProfiles3.nc
"""

"""

x0 = np.zeros((fit_data['HSRL Mol'].size+1,6))
x0[0,0] = 2.0 # HSRL Molecular
x0[0,1] = 1.0 # HSRL Combined
x0[0,2] = 1.0 # WV Online
x0[0,3] = 1.0 # WV Offline
x0[0,4] = 1.0 # O2 Online
x0[0,5] = 1.0 # O2 Offline
x0[1:,0] =  np.log(interp_vars['sim_T'])
x0[1:,1] =  np.log(interp_vars['sim_nWV']) # use water vapor data on interpolated grid to align with other data points
x0[1:,2] =  np.log(interp_vars['BSR_HSRL_Molecular'])


fit_test = td.TDProfiles(fit_data,x0.flatten(),Trx,rb_spec,abs_spec,fit_profs['WV Online'].mean_dR,inuL,multBSR,temp.profile[0,0],pres.profile[0,0],fit_profs['WV Online'].range_array[0])

phiH = 0.5*(np.log((fit_data['HSRL Mol']-fit_data['HSRL Mol BG'])/fit_test[0])+np.log((fit_data['HSRL Comb']-fit_data['HSRL Comb BG'])/fit_test[1]))
plt.figure()
plt.plot(np.log((fit_data['HSRL Mol']-fit_data['HSRL Mol BG'])/fit_test[0]),label='Molecular')
plt.plot(np.log((fit_data['HSRL Comb']-fit_data['HSRL Comb BG'])/fit_test[1]),label='Combined')
plt.plot(phiH,'--',label='mean')
plt.title('HSRL')
plt.legend()

phiW = 0.5*(np.log((fit_data['WV Online']-fit_data['WV Online BG'])/fit_test[2])+np.log((fit_data['WV Offline']-fit_data['WV Offline BG'])/fit_test[3]))
plt.figure()
plt.plot(np.log((fit_data['WV Online']-fit_data['WV Online BG'])/fit_test[2]),label='Online')
plt.plot(np.log((fit_data['WV Offline']-fit_data['WV Offline BG'])/fit_test[3]),label='Offline')
plt.plot(phiW,'--',label='mean')
plt.title('WV DIAL')
plt.legend()

phiO = 0.5*(np.log((fit_data['O2 Online']-fit_data['O2 Online BG'])/fit_test[4])+np.log((fit_data['O2 Offline']-fit_data['O2 Offline BG'])/fit_test[5]))
plt.figure()
plt.plot(np.log((fit_data['O2 Online']-fit_data['O2 Online BG'])/fit_test[4]),label='Online')
plt.plot(np.log((fit_data['O2 Offline']-fit_data['O2 Offline BG'])/fit_test[5]),label='Offline')
plt.plot(phiO,'--',label='mean')
plt.title('O2 DIAL')
plt.legend()


#K_Hmol = 1e18
#hsrl_coeff = interp_vars['Overlap']/(fit_profs['BSR'].range_array)**2*interp_vars['sim_beta_mol']*interp_vars['BSR_HSRL_Molecular']*interp_vars['Taer_HSRL_Molecular']
#
#plt.figure()
#plt.semilogx(K_Hmol*hsrl_coeff*fit_test[0],fit_profs['BSR'].range_array)
#plt.semilogx(fit_data['HSRL Mol'],fit_profs['BSR'].range_array)



isim = 1000
molBeta_nu = lp.RB_Spectrum(sim_vars['sim_T'][isim:isim+1],sim_vars['sim_P'][isim:isim+1],profs['HSRL Molecular'].wavelength,nu=sim_vars['sim_nu'],norm=True)
molPCA =td.calc_pca_spectrum(rb_spec['HSRL'],sim_vars['sim_T'][isim:isim+1],sim_vars['sim_T'][0],sim_vars['sim_P'][0])  
plt.figure()
plt.semilogy(sim_vars['sim_nu'],molBeta_nu)
plt.semilogy(nu,molPCA)

plt.figure()
plt.semilogy(sim_vars['sim_nu'],sim_vars['Tetalon_HSRL_Molecular']*sim_vars['RbFilter'])
plt.semilogy(nu,Trx['HSRL Mol'],'--')

print(np.sum(molBeta_nu.T*sim_vars['Tetalon_HSRL_Molecular']*sim_vars['RbFilter']))
print(np.sum(Trx['HSRL Mol']*molPCA.T))

ch_name = 'WV Online'
ext_wv = lp.WV_ExtinctionFromHITRAN(lp.c/profs[ch_name].wavelength+sim_vars['sim_nu'],sim_vars['sim_T'][isim:isim+1],sim_vars['sim_P'][isim:isim+1],nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9]),freqnorm=True).T
#ext_wv2 = spec.ExtinctionFromHITRAN(lp.c/profs[ch_name].wavelength+sim_vars['sim_nu'],sim_vars['sim_T'][isim:isim+1],sim_vars['sim_P'][isim:isim+1],nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9]),freqnorm=True).T
ext_wv_PCA = td.calc_pca_spectrum(abs_spec[ch_name],sim_vars['sim_T'][isim:isim+1],sim_vars['sim_T'][0],sim_vars['sim_P'][0])  

molBeta_nu = lp.RB_Spectrum(sim_vars['sim_T'][isim:isim+1],sim_vars['sim_P'][isim:isim+1],profs[ch_name].wavelength,nu=sim_vars['sim_nu'],norm=True)
molPCA =td.calc_pca_spectrum(rb_spec[ch_name],sim_vars['sim_T'][isim:isim+1],sim_vars['sim_T'][0],sim_vars['sim_P'][0])  

plt.figure()
plt.semilogy(sim_vars['sim_nu'],sim_vars['Tetalon_'+ch_name.replace(' ','_')][:,np.newaxis]*np.exp(-ext_wv*sim_vars['sim_nWV'][isim:isim+1]*100))
plt.semilogy(nu,Trx[ch_name][:,np.newaxis]*np.exp(-ext_wv_PCA*sim_vars['sim_nWV'][isim:isim+1]*100),'--')
plt.title(ch_name)

print(np.sum(molBeta_nu*sim_vars['Tetalon_'+ch_name.replace(' ','_')][:,np.newaxis]*np.exp(-ext_wv*sim_vars['sim_nWV'][isim:isim+1]*100)))
print(np.sum(Trx[ch_name][:,np.newaxis]*np.exp(-ext_wv_PCA*sim_vars['sim_nWV'][isim:isim+1]*100)*molPCA))




ch_name = 'O2 Online'
ext_o2 = spec.ExtinctionFromHITRAN(lp.c/profs[ch_name].wavelength+sim_vars['sim_nu'],sim_vars['sim_T'],sim_vars['sim_P'],(td.mO2*1e-3)/lp.N_A,nuLim=np.array([lp.c/770e-9,lp.c/768e-9]),freqnorm=True,filename=o2_spec_file).T
OD_o2 = np.cumsum(sim_vars['sim_nO2'][np.newaxis,:]*ext_o2,axis=1)*sim_vars['sim_dr']


convP = 101325.0  # pressure conversion factor for O2 number density calculation
P = convP*sim_vars['sim_P'][0]*(interp_vars['sim_T']/sim_vars['sim_T'][0])**5.2199 

dr_int = np.ones((1,interp_vars['sim_nWV'].size))*fit_profs['BSR'].mean_dR
#dr_int[0] = r0
sigS = td.calc_pca_spectrum(abs_spec[ch_name],interp_vars['sim_T'],sim_vars['sim_T'][0],sim_vars['sim_P'][0])
nO2 = td.fo2*(P/(lp.kB*interp_vars['sim_T'])-interp_vars['sim_nWV'])
ODs = np.cumsum(dr_int*sigS*nO2[np.newaxis,:],axis=1)
#Ts = np.exp(-ODs)

tx_diff = np.zeros(fit_profs['BSR'].range_array.size)
for ai in range(fit_profs['BSR'].range_array.size):
    plot_alt = fit_profs['BSR'].range_array[ai]
    iplt1 = np.argmin(np.abs(sim_vars['sim_range']-plot_alt))
    iplt2 = np.argmin(np.abs(fit_profs['BSR'].range_array-plot_alt))
    
    molBeta_nu = lp.RB_Spectrum(sim_vars['sim_T'][isim:isim+1],sim_vars['sim_P'][iplt1:iplt1+1],profs[ch_name].wavelength,nu=sim_vars['sim_nu'],norm=True)
    molPCA =td.calc_pca_spectrum(rb_spec[ch_name],interp_vars['sim_T'][iplt2:iplt2+1],sim_vars['sim_T'][0],sim_vars['sim_P'][0])  
    
    tx_diff[ai] = np.sum(molBeta_nu*sim_vars['Tetalon_'+ch_name.replace(' ','_')][:,np.newaxis]*np.exp(-OD_o2[:,iplt1:iplt1+1]))-np.sum(Trx[ch_name][:,np.newaxis]*np.exp(-ODs[:,iplt2:iplt2+1])*molPCA)

plt.figure()
plt.plot(tx_diff,fit_profs['BSR'].range_array)
    
plot_alt = 4e3 
iplt1 = np.argmin(np.abs(sim_vars['sim_range']-plot_alt))
iplt2 = np.argmin(np.abs(fit_profs['BSR'].range_array-plot_alt))
plt.figure()
plt.semilogy(sim_vars['sim_nu'],ext_o2[:,iplt1],label='HITRAN')
plt.semilogy(nu,sigS[:,iplt2],'--',label='PCA')
plt.title('Extinction Coefficient at %d m'%plot_alt)
plt.legend()
plt.xlabel('Frequency [Hz]')

plt.figure()
plt.semilogy(sim_vars['sim_nu'],OD_o2[:,iplt1],label='HITRAN')
plt.semilogy(nu,ODs[:,iplt2],'--',label='PCA')
plt.title(ch_name+' Optical Depth at %d m'%plot_alt)
plt.legend()
plt.xlabel('Frequency [Hz]')

plt.figure()
plt.plot(sim_vars['sim_nu'],sim_vars['Tetalon_'+ch_name.replace(' ','_')]*np.exp(-OD_o2[:,iplt1]),label='HITRAN')
plt.plot(nu,Trx[ch_name]*np.exp(-ODs[:,iplt2]),'--',label='PCA')
plt.title(ch_name + ' Total Transmission at %d m'%plot_alt)
plt.legend()
plt.xlabel('Frequency [Hz]')

print(np.sum(molBeta_nu*sim_vars['Tetalon_'+ch_name.replace(' ','_')][:,np.newaxis]*np.exp(-OD_o2[:,iplt1:iplt1+1])))
print(np.sum(Trx[ch_name][:,np.newaxis]*np.exp(-ODs[:,iplt2:iplt2+1])*molPCA))

molPCA =td.calc_pca_spectrum(rb_spec['O2 Online'],interp_vars['sim_T'],sim_vars['sim_T'][0],sim_vars['sim_P'][0]) 

plt.figure()
plt.semilogx(sim_vars['T_tx_O2_Online'],sim_vars['sim_range'])
plt.semilogx(np.exp(-ODs[inuL['O2 Online'],:]),fit_profs['BSR'].range_array,'--')

plt.figure()
plt.semilogx(sim_vars['T_rx_aer_O2_Online'],sim_vars['sim_range'])
plt.semilogx(Trx[ch_name][inuL['O2 Online']]*np.exp(-ODs[inuL['O2 Online'],:]),fit_profs['BSR'].range_array,'--')

plt.figure()
plt.semilogx(sim_vars['T_rx_mol_O2_Online'],sim_vars['sim_range'])
plt.semilogx(np.sum(Trx[ch_name][:,np.newaxis]*np.exp(-ODs)*molPCA,axis=0),fit_profs['BSR'].range_array,'--')