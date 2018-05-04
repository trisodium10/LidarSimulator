# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 20:27:34 2017

@author: mhayman
"""

import numpy as np
import LidarProfileFunctions as lp
#import WVProfileFunctions as wv
#import FourierOpticsLib as FO
#import RbSpectrumLib as rb

import TDRetrievalLib as td

import SpectrumLib as spec

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#import datetime
#
import timeit


def TP(spec_data,T,P,nu=np.array([])):
    """
    Calculates extinction spectrum and its temperature derivative for T and P 
    arrays (of equal size)
    spec_data - dict of PCA spectral data returned by load_spect_params
    if nu is unassigned, the data is returned at native resolution
        (this is the fasted option)
    if nu is defined, the data is interpolated to the requested grid
    
    If speed is needed it is not recomended that nu be provided.  Instead
    provide the required frequency grid to load_spect_params.  That will
    configure the PCA matrices so no interpolation is required.
    """
    TPmat = np.matrix(((T.flatten()-spec_data['Tmean'])/spec_data['Tstd'])[:,np.newaxis]**spec_data['pT'] \
        *((P.flatten()-spec_data['Pmean'])/spec_data['Pstd'])[:,np.newaxis]**spec_data['pP'])
    return TPmat

def TP_deriv(spec_data,T,P,nu=np.array([])):
    """
    Calculates extinction spectrum and its temperature derivative for T and P 
    arrays (of equal size)
    spec_data - dict of PCA spectral data returned by load_spect_params
    if nu is unassigned, the data is returned at native resolution
        (this is the fasted option)
    if nu is defined, the data is interpolated to the requested grid
    
    If speed is needed it is not recomended that nu be provided.  Instead
    provide the required frequency grid to load_spect_params.  That will
    configure the PCA matrices so no interpolation is required.
    """
#    TPmat = np.matrix(((T.flatten()-spec_data['Tmean'])/spec_data['Tstd'])[:,np.newaxis]**spec_data['pT'] \
#        *((P.flatten()-spec_data['Pmean'])/spec_data['Pstd'])[:,np.newaxis]**spec_data['pP'])
    dTPmat = np.matrix(spec_data['pT']*(((T.flatten()-spec_data['Tmean'])/spec_data['Tstd'])[:,np.newaxis]**(spec_data['dpT'])) \
        *(((P.flatten()-spec_data['Pmean'])/spec_data['Pstd'])[:,np.newaxis]**spec_data['pP'])/spec_data['Tstd'])
        
    return dTPmat


pca_file_path = '/Users/mhayman/Documents/DIAL/PCA_Spectroscopy/'

wavelength_list = np.array([828.203e-9,828.3026e-9,769.2339e-9,769.319768e-9])  # offline,online,Comb,Mol,online,offline
name_list = ['WV Online','WV Offline','O2 Online','O2 Offline']  # name corresponding to each wavelength
index_list = np.arange(len(wavelength_list))
s_i = np.array([0,0,1,1])  # index into particular species definition

species_name =  ['H2O','O2']
species_mass_list = np.array([lp.mH2O,spec.mO2])
spec_file = ['/Users/mhayman/Documents/DIAL/WV_HITRAN2012_815_841.txt','/Users/mhayman/Documents/DIAL/O2_HITRAN2012_760_781.txt']
spec_range = [np.array([lp.c/828.5e-9,lp.c/828e-9]),np.array([lp.c/770e-9,lp.c/768e-9])]

pca_files = ['PCA_Data_WV_Online_828203pm.npz','PCA_Data_WV_Offline_828303pm.npz','PCA_Data_O2_Online_769234pm.npz','PCA_Data_O2_Offline_769320pm.npz']

sim_i = 2
#tH_exec = []
#t_exec = []
#n_exec = []

Pres = np.array([0.9])
Temp = np.array([312.0])
#Pres = np.linspace(0.4,1.0,1)
#Temp = np.linspace(210,290,Pres.size)

sim_dnu = 20e6  # spacing in optical frequency space of simulation
sim_nu_max = 3e9  # edge of simulated frequency space
#sim_range = np.arange(0,15e3,sim_dr)  # range array in m
sim_nu = np.arange(-sim_nu_max,sim_nu_max,sim_dnu) # frequency array in Hz

start_time = timeit.default_timer()
#ext_hitran = spec.ExtinctionFromHITRAN(lp.c/wavelength_list[sim_i]+sim_nu,Temp.flatten(),Pres.flatten(),(species_mass_list[s_i[sim_i]]*1e-3)/lp.N_A,nuLim=spec_range[s_i[sim_i]],freqnorm=True,filename=spec_file[s_i[sim_i]]).T
ext_hitran = spec.ExtinctionFromHITRAN(lp.c/wavelength_list[sim_i]+sim_nu,Temp.flatten(),Pres.flatten(),species_name[s_i[sim_i]],nuLim=spec_range[s_i[sim_i]],freqnorm=True).T
elapsed = timeit.default_timer() - start_time

pca_file = spec.get_pca_filename('abs',wavelength=wavelength_list[sim_i],name=name_list[sim_i])
spec_params = spec.load_spect_params(pca_file)
spec_params_int = spec.load_spect_params(pca_file,nu=sim_nu)
#spec_params = spec.load_spect_params(pca_file_path+pca_files[sim_i])
#spec_params_int = spec.load_spect_params(pca_file_path+pca_files[sim_i],nu=sim_nu)

start_time = timeit.default_timer()
#ext_pca = td.calc_pca_spectrum(spec_params,Temp,Pres,nu=sim_nu)
ext_pca = spec.calc_pca_spectrum(spec_params,Temp,Pres)
elapsed2 = timeit.default_timer() - start_time
ext_pca_int = spec.calc_pca_spectrum(spec_params_int,Temp,Pres)

print('t HITRAN: %f'%elapsed)
print('    per point: %f'%(elapsed/Pres.size))
print('t PCA: %f'%elapsed2)
print('    per point: %f'%(elapsed2/Pres.size))

#tH_exec.extend([elapsed])
#t_exec.extend([elapsed2])
#n_exec.extend([Temp.size])

plt.figure()
plt.plot(sim_nu*1e-9,ext_hitran)
#plt.plot(sim_nu*1e-9,ext_pca,'--')
plt.plot(spec_params['nu_pca']*1e-9,ext_pca,'--')
plt.plot(spec_params_int['nu_pca']*1e-9,ext_pca_int,':')
plt.grid(b=True)
plt.ylabel('Extinction Cross Section [$m^2$]')
plt.xlabel('Frequency [GHz]')

# analytical derivatives from Binietoglou et. al, AO 2016
#nu_rb = nu=sim_nu
## test derivative operations for RB spectrum
#rb_spec = spec.load_RB_params(filename='/Users/mhayman/Documents/Python/Lidar/LidarSimulator/RB_PCA_Params.npz')
#rb,drb,rb_x = spec.calc_pca_RB_w_deriv(rb_spec,wavelength_list[sim_i],Temp,Pres)
#rb_i,drb_i = spec.calc_pca_RB_w_deriv(rb_spec,wavelength_list[sim_i],Temp,Pres,nu=nu_rb)
#fun_rb = lambda x: spec.calc_pca_RB(rb_spec,wavelength_list[sim_i],x,Pres,nu=nu_rb,norm=True)
#drbNum = td.Num_Jacob(fun_rb,Temp,step_size=1e-12)
#plt.figure(); 
#plt.plot(nu_rb*1e-9,drb_i); 
#plt.plot(nu_rb*1e-9,drbNum,'--')
##plt.semilogy(spec_params_int['nu_pca']*1e-9,dext_i,':'); 
#
#dMn = np.diff(rb_spec['M'],axis=0)/np.mean(np.diff(rb_spec['x'].flatten()))
#x_dMn = 0.5*(rb_spec['x'][:-1]+rb_spec['x'][1:])
#plt.figure(); 
#plt.plot(x_dMn,dMn[:,0]); 
#plt.plot(rb_spec['x'],rb_spec['dM'][:,0])
#
#fun_x = lambda x: lp.RayleighBrillouin_X(x,wavelength_list[sim_i],nu_rb)
#dxN = td.Num_Jacob(fun_x,Temp,step_size=1e-3)
#dxA = -0.5*lp.RayleighBrillouin_X(Temp,wavelength_list[sim_i],nu_rb)/Temp
#plt.figure()
#plt.plot(nu_rb*1e-9,dxA)
#plt.plot(nu_rb*1e-9,dxN,'--')

nu_rb = nu=sim_nu
# test derivative operations for RB spectrum
rb_fn = spec.get_pca_filename('RB',name='HSRL')
rb_spec = spec.load_spect_params(rb_fn,nu=nu_rb)
rb,drb = spec.calc_pca_spectrum_w_deriv(rb_spec,Temp,Pres)
fun_rb = lambda x: spec.calc_pca_spectrum(rb_spec,x,Pres)
drbNum = td.Num_Jacob(fun_rb,Temp,step_size=1e-12)
plt.figure(); 
plt.plot(nu_rb*1e-9,drb); 
plt.plot(nu_rb*1e-9,drbNum,'--')
#plt.semilogy(spec_params_int['nu_pca']*1e-9,dext_i,':'); 


# test derivative operations
ext,dext = spec.calc_pca_spectrum_w_deriv(spec_params,Temp,Pres)
ext_i,dext_i = spec.calc_pca_spectrum_w_deriv(spec_params_int,Temp,Pres)
fun_ext = lambda x: spec.calc_pca_spectrum(spec_params,x,Pres)
dextNum = td.Num_Jacob(fun_ext,Temp,step_size=1e-2)
#fun_ext_hi = lambda x: spec.ExtinctionFromHITRAN(lp.c/wavelength_list[sim_i]+spec_params_int['nu_pca'],x,Pres.flatten(),(species_mass_list[s_i[sim_i]]*1e-3)/lp.N_A,nuLim=spec_range[s_i[sim_i]],freqnorm=True,filename=spec_file[s_i[sim_i]]).T
fun_ext_hi = lambda x: spec.ExtinctionFromHITRAN(lp.c/wavelength_list[sim_i]+spec_params_int['nu_pca'],x,Pres.flatten(),species_name[s_i[sim_i]],nuLim=spec_range[s_i[sim_i]],freqnorm=True).T
dext_hiNum = td.Num_Jacob(fun_ext_hi,Temp,step_size=1e-2)
plt.figure(); 
plt.plot(spec_params['nu_pca']*1e-9,dext,label='PCA, native grid'); 
plt.plot(spec_params['nu_pca']*1e-9,dextNum,'--',label='Numerical PCA, native grid')
plt.plot(spec_params_int['nu_pca']*1e-9,dext_i,'r:',label='PCA, custom grid'); 
plt.plot(spec_params_int['nu_pca']*1e-9,dext_hiNum,'k:',label='Numemrical HITRAN')
plt.legend()

dTP = TP_deriv(spec_params_int,Temp,Pres)
fun_TP = lambda x: TP(spec_params,x,Pres)
dTPNum = td.Num_Jacob(fun_TP,Temp,step_size=1e-15)
plt.figure()
plt.semilogy(np.array(dTP).flatten())
plt.semilogy(dTPNum.flatten(),'--')

