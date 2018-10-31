#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:40:53 2018

@author: mhayman
"""


import numpy as np
import matplotlib.pyplot as plt

import TDMPDLib as td

import LidarProfileFunctions as lp
import LidarPlotFunctions as lplt
import MLELidarProfileFunctions as mle
import SpectrumLib as spec

import datetime

import netCDF4 as nc4

import json

import copy

import os

#import pickle

#ncpath = '/h/eol/mhayman/DIAL/Processed_Data/'
ncpath = '/Users/mhayman/Documents/DIAL/Processed_Data/'

# high time resolution
nclist0 = ['wv_dial05.181020.Python.nc']
t_start = 4*3600+0*60  # start time in seconds
t_stop = 5*3600+0*60  # stop time in seconds
t_duration = 60*60  # duration of a single denoising step
t_increment = 60*60  # time increment after each denoising step

r_lim = [0.3e3,4e3]
step_eps = 1e-7 #1e-7 #2e-5
max_iter = 1000 #1500
deconv = True
lam_set = {'xB':30.0,'xN':49.8,'xT':150.0,'xPhi':64.0,'xPsi':64.0}
#lam_set = {'xB':0,'xN':0,'xT':0,'xPhi':0,'xPsi':0} # for validating gradient function
lam_range0 = {'xB':[0,2],'xN':[1.0,2.0],'xT':[2.0,3.0],'xPhi':[1.0,2.0],'xPsi':[1.0,2.0]}  # note these search bounds are log10
#Num_reg_iter = 1
verbose = True
plot_results = False
show_plots = True
opt_setting = {'Num_reg_first':10,  # number of times to evaluate the regularizer during the first profile
               'Num_reg_next':5,    # number of times to evaluate the regularizer on each subsequent profile
               'reg_next_factor':0.1  # adjustment factor for regularizer range on all subsequent evaluations
               }


var_scale = {'xB':1,'xN':1,'xT':1,'xPhi':1,'xPsi':1}

# if desired, loaded files will be constrainted by time
time_start = datetime.datetime(year=1900,month=1,day=1)
time_stop = datetime.datetime(year=2100,month=1,day=1)


load_mask = False
save_results = False

# set the variable bounds in their native space.
# this is converted to the state variable later
# This variable is not actually used
opt_bounds = {'xG':[-1e10,1e10],
              'xB':[-1e30,1e10],
              'xN':[-1e10,1e10],
              'xT':[0,500],
              'xPhi':[-1e10,1e10],
              'xPsi':[-1e10,1e10]}

#prof_list = ['Aerosol_Backscatter_Coefficient','Aerosol_Extinction_Coefficient','Merged_Combined_Channel',
#             'Particle_Depolarization','Volume_Depolarization','Denoised_Aerosol_Backscatter_Coefficient'] #,'Denoised_Aerosol_Backscatter_Coefficient'] #'Aerosol_Extinction_Coefficient'

#prof_list = ['Absolute_Humidity',
#             'Online_Backscatter_Channel_Raw_Data',
#             'Offline_Backscatter_Channel_Raw_Data',
#             'Temperature','Pressure','Water_Vapor_Range_Resolution','Denoised_Aerosol_Backscatter_Coefficient']
             
prof_list = ['Absolute_Humidity',
             'WV_Online_Backscatter_Channel_Raw_Data',
             'WV_Offline_Backscatter_Channel_Raw_Data',
             'O2_Online_Backscatter_Channel_Raw_Data',
             'O2_Offline_Backscatter_Channel_Raw_Data',
             'O2_Online_Molecular_Detector_Backscatter_Channel_Raw_Data',
             'O2_Offline_Molecular_Detector_Backscatter_Channel_Raw_Data',
             'Temperature','Pressure','Backscatter_Ratio',
             'Surface_Temperature_HSRL',
             'Surface_Pressure_HSRL']
# new names for the profiles
prof_name = ['Absolute_Humidity','WVOnline','WVOffline','CombOnline','CombOffline','MolOnline','MolOffline','Temperature','Pressure','Backscatter_Ratio','Surf_T','Surf_P']
#  Naming convention list:
# 'WVOnline', 'WVOffline', 'MolOnline', 'MolOffline', 'CombOnline', 'CombOffline'

lidar_vars = ['WVOnline_wavelength','WVOffline_wavelength','O2Online_wavelength','O2Offline_wavelength']

t_start_list = np.arange(t_start,t_stop,t_increment)

print('Expecting %d denoising steps'%t_start_list.size)  

#%%  Load Data Files
    
for denoise_step in range(t_start_list.size):
    # set time limits on this denoised profile
    t_lim = [t_start_list[denoise_step],min([t_start_list[denoise_step]+t_duration,t_stop])]
    
    print('')
    print('(%d / %d)  Denoising '%(denoise_step+1,t_start_list.size)+'%.2f h-%.2f h'%(t_lim[0]/3600.0,t_lim[1]/3600.0))


    if denoise_step == 0:
        Num_reg_iter = opt_setting['Num_reg_first']
        if Num_reg_iter > 1:
            lam_range = copy.deepcopy(lam_range0)
    else:
        Num_reg_iter = opt_setting['Num_reg_next']
        if Num_reg_iter > 1:
            lam_range = {}
            for xvar in lam_range0.keys():
                lam_range[xvar] = [np.log10(lam_set[xvar])-0.5*opt_setting['reg_next_factor']*(lam_range0[xvar][1]-lam_range0[xvar][0]),\
                                    np.log10(lam_set[xvar])+0.5*opt_setting['reg_next_factor']*(lam_range0[xvar][1]-lam_range0[xvar][0])]
                print('lambda space for ' + xvar+ ': 10^ %f to %f'%(lam_range[xvar][0],lam_range[xvar][1]))

    """
    Load selected data files
    """
    
    # sort the files by name and add the path to the files
    nclist = [ncpath+x for x in sorted(nclist0)]
    
    
    findex = []
    aircraft_data = {}
    lidar_data = {}
    
    # Code to constrain the files by time.
    for ai in range(len(nclist)):
    #    filedate = datetime.datetime.strptime(nclist[ai][-24:],'wv_dial.%y%m%d.Python.nc')
        filedate = datetime.datetime.strptime(nclist[ai][-17:],'.%y%m%d.Python.nc')
    
        if time_start <= filedate and time_stop >= filedate:
            findex.extend([ai])
        elif time_start <= filedate+datetime.timedelta(days=1) and time_stop >= filedate+datetime.timedelta(days=1):
            findex.extend([ai])
        
    
    
    prof_start = True
    profs = {}
    aircraft_data = {}
    lidar_data = {}
    for fi in findex:
        ncfilename = nclist[fi]
        f = nc4.Dataset(ncfilename,'r')     
        
        # load lidar variables
        for l_var in lidar_vars:
            new_data = lp.ncvar(f,l_var)
            if l_var in lidar_data.keys():                
                if new_data.ndim == 2:
                    lidar_data[l_var] = np.hstack((lidar_data[l_var],new_data))
                else:
                    lidar_data[l_var] = np.concatenate((lidar_data[l_var],new_data))
            else:
                lidar_data[l_var] = new_data
                
    
        f.close()

        
        for ivar,var in enumerate(prof_list):
            prof0 = lp.load_nc_Profile(ncfilename,var,mask=load_mask)
    	    # check if the load was successful
            if hasattr(prof0,'time'):
                if prof_name[ivar] in profs.keys():
                    profs[prof_name[ivar]].cat_time(prof0,front=False)
                else:
                    profs[prof_name[ivar]] = prof0.copy()


    #%% configure save path and file names
    var = list(profs.keys())[0]


    save_start_time = profs[var].StartDate+datetime.timedelta(microseconds=np.int(1e6*profs[var].time[0]))
    save_stop_time = profs[var].StartDate+datetime.timedelta(microseconds=np.int(1e6*profs[var].time[-1]))
    if save_results:
        save_path = profs[var].StartDate.strftime('/Users/mhayman/Documents/DIAL/plots/Denoise_%Y%m%d/')
        if not os.path.exists(save_path):
                    os.makedirs(save_path)
        save_file=profs[var].lidar.replace(' ','').replace('-','') + '_SubDenoise_'+save_start_time.strftime('%Y%m%dT%H%M%S')+'_'+save_stop_time.strftime('%Y%m%dT%H%M%S')
#        save_file = profs[var].StartDate.strftime('%Y%m%d')+'_%d_%d_UTC_'%(t_lim[0]/3600,t_lim[1]/3600)+datetime.datetime.now().strftime('%Y%m%d_%H%M')
        savefigpath = save_path + save_file
        savencfile = save_file +'.nc'


    #    if save_results:
    #        save_path = profs[var].StartDate.strftime('/Users/mhayman/Documents/DIAL/plots/Denoise_%Y%m%d/')
    #        if not os.path.exists(save_path):
    #                    os.makedirs(save_path)
    #        save_file = profs[var].StartDate.strftime('%Y%m%d')+'_%d_%d_UTC_'%(t_lim[0]/3600,t_lim[1]/3600)+datetime.datetime.now().strftime('%Y%m%d_%H%M')
    #        if deconv:
    #            save_file = save_file + 'deconv'
    #    #    save_path = '/Users/mhayman/Documents/DIAL/plots/Denoise_20180519/'
    #    #    save_file = '20170818_4UTC_20180519'
                
    #%%  Plot Sample Profile
    
    #var = 'Absolute_Humidity'
    #var = 'Offline'
    #var = 'Aerosol_Backscatter_Coefficient'
    #lplt.pcolor_profiles([profs[var]],cmap=[plot_settings[var]['colormap']],scale=[plot_settings[var]['scale']],
    #                     climits=[plot_settings[var]['climits']],plot_date=True)
                     
    #%%  Trim and prep data for processing

    """
    Prep data for processing
    """
    
    #def MLE_Cals_2D(MolRaw,CombRaw,beta_aer,surf_temp,surf_pres,geo_data,minSNR=0.5,\
    #t_lim=np.array([np.nan,np.nan]),verify=False,verbose=False,print_sol=True,\
    #plotfigs=True,lam_array=np.array([np.nan]),Nmax=2000):
    """
    Runs a maximum likelihood estimator to correct for
        Channel gain mismatch, 
        Aerosol to Molecular Channel Crosstalk
        Detector Dead Time
    and obtain estimates of
        Backscatter coefficient
        Extinction coefficient
        Lidar Ratio
        Each Channel's gain
        Aerosol to Molecular Channel Crosstalk
        Detector Dead Time
        
    Inputs:
        MolRaw - Raw Molecular Profile
        CombRaw - Raw Combined Profile
        beta_aer - derived estimate of aerosol backscatter coefficent
        surf_temp - temperature data from the surface station
        surf_pres - pressure data from the surface station
        geo_data - data file from loading the geometric overlap function
        minSNR - minimum SNR required to be included as containing possible aerosols
        t_lim - currently disabled.  when added, will allow us to select only a segment to operate on.
        verify - if True it will use Poisson Thinning to verify TV solution
        verbose - if True it will output the fit result of each TV iteration
        print_sol - print the solved calibration parameters
        plotfigs - plot the TV verification error
        lam_array - array containing TV values to be evaluated
        Nmax - maximum optimizer iterations
        
    Substitutions:
        beta_aer for aer_beta_dlb
        beta_aer_E for aer_beta_E
    """
    
    # Trim the data in time and range
    
    var = 'WVOnline'
    t_data = profs[var].time.copy()  # store time data for 1d variables
    
    raw_range = profs[var].range_array.copy()  # used to trim the geo files later
    
    #t_lim = [26*3600+13*60,26*3600+13.4*60] #
    #r_lim = [0,2e3]
    
    # force range limits to as or more limiting than most restrictive profile
    for var in profs.keys():
        if not 'Surf' in var:
            if r_lim[0] < profs[var].range_array[0]:
                r_lim[0] = profs[var].range_array[0]
            if r_lim[1] > profs[var].range_array[-1]:
                r_lim[1] = profs[var].range_array[-1]
    
    # additional processing for the raw profiles:
    # 1.   Estimate background, but don't subtract it
    # 2.   Align the range grid with process profiles
    for var in profs.keys():
        profs[var].slice_time(t_lim)
        if not 'Surf' in var:
            if 'line' in var:
                # run a custom background estimation
                # store the result in the profile background (bg) but don't
                # actually subtract it
                profs[var].bg = np.nanmean(profs[var].profile[:,-100:],axis=1)
                profs[var].bg_var = np.nanmean(profs[var].profile[:,-100:],axis=1)
                # now trim the range dimension to match the processed profiles
        #        profs[var].slice_range(range_lims=[profs['Aerosol_Backscatter_Coefficeint'].range_array[0],profs['Aerosol_Backscatter_Coefficeint'].range_array[1]])
            profs[var].slice_range(range_lim=r_lim)
    
    
    # trim time on 1D variables
    irm1d = np.nonzero((t_data < t_lim[0]) +(t_data > t_lim[1]))
    for var1d in aircraft_data.keys():
        aircraft_data[var1d] = np.delete(aircraft_data[var1d],irm1d)
    for var1d in lidar_data.keys():
        if lidar_data[var1d].ndim > 1:
            lidar_data[var1d] = np.delete(lidar_data[var1d],irm1d,axis=1)
        else:
            lidar_data[var1d] = np.delete(lidar_data[var1d],irm1d)
        
    # copy the raw profiles to avoid modifying them
    raw_profs = {}
    for var in profs.keys():
        if 'line' in var:
            raw_profs[var] = profs[var].copy()
            raw_profs[var].profile = raw_profs[var].profile*raw_profs[var].NumProfList[:,np.newaxis]
            raw_profs[var].bg = raw_profs[var].bg*raw_profs[var].NumProfList
    LidarNumber = np.int(profs['CombOnline'].lidar[-1])

    #%%  Load cals and trim them
    """
    Load Calibration Parameters
    """
    
    #cal_file_path = '/h/eol/mhayman/PythonScripts/HSRL_Processing/GV-HSRL-Python/calibrations/cal_files/'
    cal_file_path = '/Users/mhayman/Documents/Python/Lidar/NCAR-LidarProcessing/calibrations/'
    #cal_file = cal_file_path+'dial1_calvals.json'
    cal_file = cal_file_path+'dial%d_calvals.json'%LidarNumber
    
    with open(cal_file,"r") as f:
        cal_json = json.loads(f.read())
    f.close()
    
    # This isn't that sophisticated, but it works.  Don't try to update the time_start variable
    time_start = profs['WVOnline'].StartDate
    
    # Load the appropriate calibrations
    #dead_time_list = lp.get_calval(time_start,cal_json,"Dead_Time",returnlist=['combined_hi','cross','combined_lo','molecular'])
    #dead_time = dict(zip(['High_Gain_Total_Backscatter_Channel','Cross_Polarization_Channel','Low_Gain_Total_Backscatter_Channel','Molecular_Backscatter_Channel'],dead_time_list))
    wv_sigma = {}
    for var in raw_profs.keys():
        if 'WV' in var:
            sig_file = sig_on_fn = spec.get_pca_filename('Abs',name='WV '+str(var))
            sig_data = spec.load_spect_params(sig_file,wavelen=np.array([raw_profs[var].wavelength]))
            


#        wv_sigma[var] = spec.calc_pca_spectrum(sig_data,profs['Temperature'].profile.flatten(),profs['Pressure'].profile.flatten()).reshape(profs['Temperature'].profile.shape)
        
#        sig_file = sig_on_fn = spec.get_pca_filename('Abs',name='WV '+str(var))
#        sig_data = spec.load_spect_params(sig_file,wavelen=np.array([raw_profs[var].wavelength]))
#        wv_sigma[var] = spec.calc_pca_spectrum(sig_data,profs['Temperature'].profile.flatten(),profs['Pressure'].profile.flatten()).reshape(profs['Temperature'].profile.shape)
        
    ## load the PCA spectroscopy files and data          
    #sig_on_fn = spec.get_pca_filename('Abs',name='WV Online')
    #sig_on_data = spec.load_spect_params(sig_on_fn,wavelen=raw_profs['Online'].wavelength)
    #sig_on = spec.calc_pca_spectrum(sig_on_data,profs['Temperature'].profile.flatten(),profs['Pressure'].profile.flatten())
    #
    #sig_off_fn = spec.get_pca_filename('Abs',name='WV Offline')
    #sig_off_data = spec.load_spect_params(sig_off_fn,wavelen=raw_profs['Offline'].wavelength)
    #sig_off = spec.calc_pca_spectrum(sig_off_data,profs['Temperature'].profile.flatten(),profs['Pressure'].profile.flatten())
    #  
    #sig_off = sig_on.reshape(profs['Temperature'].shape) 
    #sig_off = sig_off.reshape(profs['Temperature'].shape)
    
    
    # load the expected molecular backscatter
    beta_m = lp.get_beta_m(profs['Temperature'],profs['Pressure'],raw_profs['WVOnline'].wavelength)
    
    # load geofile from HSRL just to get a rough idea for the fitting routine
    geo_file = lp.get_calval(time_start,cal_json,"Geo_File_Record",returnlist=['filename'])
    geo_data = np.load(cal_file_path+geo_file[0])
    
    geo_func = np.interp(raw_profs['WVOnline'].range_array,geo_data['geo_prof'][:,0],1.0/geo_data['geo_prof'][:,1])
    
    # load diff_geofile from HSRL
    diff_geo_file = lp.get_calval(time_start,cal_json,"Molecular Gain",cond=[['diff_geo','!=','none']],returnlist=['value','diff_geo'])
    diff_geo_data = np.load(cal_file_path+diff_geo_file[1])
    
    diff_geo_func = np.interp(raw_profs['WVOnline'].range_array,diff_geo_data['range_array'],diff_geo_data['diff_geo_prof'].flatten())
    
    iso_list = ['39','40','41']
    CellTemp = lp.get_calval(time_start,cal_json,'Gas Cell Temperature')[0]
    CellLength = lp.get_calval(time_start,cal_json,'Gas Cell Length')[0]
    CellPressure = lp.get_calval(time_start,cal_json,'Gas Cell Pressure')[0]
    CellPurity = lp.get_calval(time_start,cal_json,'Gas Cell Isotope Purity')[0]
    CellSpecies = lp.get_calval(time_start,cal_json,'Gas Cell Species')[0]
    
    ## setup variable geo overlap (up vs down pointing) if supplied
    #if len(geo_file_down) > 0:
    #    geo_data = {}
    #    key_list = ['geo_mol','geo_mol_var','Nprof']
    #    for var in key_list:
    #        if var in geo_up.keys():
    #            geo_data[var] = np.ones((lidar_data['TelescopeDirection'].size,profs['Raw_Molecular_Backscatter_Channel'].range_array.size))
    #            geo_data[var][np.nonzero(lidar_data['TelescopeDirection']==1.0)[0],:] = np.interp(profs['Raw_Molecular_Backscatter_Channel'].range_array,raw_range,geo_up[var])
    #            if var in geo_down.keys():
    #                geo_data[var][np.nonzero(lidar_data['TelescopeDirection']==0.0)[0],:] = np.interp(profs['Raw_Molecular_Backscatter_Channel'].range_array,raw_range,geo_down[var])
    #            else:
    #                geo_data[var][np.nonzero(lidar_data['TelescopeDirection']==0.0)[0],:] = np.interp(profs['Raw_Molecular_Backscatter_Channel'].range_array,raw_range,geo_up[var])
    #        else:
    #            geo_data[var] = np.ones((lidar_data['TelescopeDirection'].size,1))


    #%% Build dictionary of constant range multipliers
    nu_pca = np.linspace(-3e9,3e9,41)
    thin_adj = 0.5
    
    ConstTerms = {}
    for var in raw_profs.keys():
        ConstTerms[var] = {'mult':np.zeros(raw_profs[var].profile.shape),'Trx':np.ones(nu_pca.shape),'bg':np.zeros((raw_profs[var].time.size,1))}
        ConstTerms[var]['mult'] = raw_profs[var].NumProfList[:,np.newaxis]*geo_func[np.newaxis,:]*beta_m.profile/raw_profs[var].range_array[np.newaxis,:]**2
        if 'Comb' in var:
            ConstTerms[var]['mult']*=diff_geo_func
        ConstTerms[var]['bg'] = raw_profs[var].bg[:,np.newaxis]
        if 'MolOffline' in var:   
            ConstTerms[var]['Trx'] = spec.CellTransmission(nu_pca+lp.c/raw_profs[var].wavelength,CellTemp,CellPressure,CellLength,spec.KD1defs,iso=iso_list)
        ConstTerms[var]['rate_adj'] = (thin_adj*1.0/(raw_profs[var].shot_count*raw_profs[var].binwidth_ns*1e-9))[:,np.newaxis]

    ConstTerms['dR'] = raw_profs[var].mean_dR
    
    if deconv:
        # add laser pulse
        ConstTerms['kconv'] = np.ones((1,4))*0.25
    
    ConstTerms['base_P'] = profs['Surf_P'].profile.flatten()
    ConstTerms['base_T'] = profs['Surf_T'].profile.flatten()
    
    
    ConstTerms['absPCA'] = {}
    ConstTerms['molPCA'] = {}
    for var in raw_profs.keys():
        if 'WVOnline' in var:
            sig_file = spec.get_pca_filename('Abs',wavelength=raw_profs[var].wavelength) #name=str(var))
            ConstTerms['absPCA']['WVon'] = spec.load_spect_params(sig_file,wavelen=np.array([raw_profs[var].wavelength]))
        elif 'WVOffline' in var:
            sig_file = spec.get_pca_filename('Abs',wavelength=raw_profs[var].wavelength)
            ConstTerms['absPCA']['WVoff'] = spec.load_spect_params(sig_file,wavelen=np.array([raw_profs[var].wavelength]))
        elif 'CombOnline' in var:
            sig_file = spec.get_pca_filename('Abs',wavelength=raw_profs[var].wavelength)
            ConstTerms['absPCA']['O2on'] = spec.load_spect_params(sig_file,wavelen=raw_profs[var].wavelength,nu=nu_pca)
            sig_file = spec.get_pca_filename('RB',wavelength=raw_profs[var].wavelength)
            ConstTerms['molPCA']['O2'] = spec.load_spect_params(sig_file,nu=nu_pca,normalize=True)
        elif 'CombOffline' in var:
            sig_file = spec.get_pca_filename('Abs',wavelength=raw_profs[var].wavelength)
            ConstTerms['absPCA']['O2off'] = spec.load_spect_params(sig_file,wavelen=np.array([raw_profs[var].wavelength]))

    ConstTerms['i0'] = np.argmin(np.abs(ConstTerms['molPCA']['O2']['nu_pca']))
    
    
    """
    # Test code
    T = profs['Temperature'].profile
    tau_on = spec.calc_pca_T_spectrum(ConstTerms['absPCA']['WVon'],T,ConstTerms['base_T'],ConstTerms['base_P'])
    tau_on = tau_on.reshape(profs['Absolute_Humidity'].profile.shape)
    
    tau_off = spec.calc_pca_T_spectrum(ConstTerms['absPCA']['WVoff'],T,ConstTerms['base_T'],ConstTerms['base_P'])  # water vapor absorption cross section
    tau_off = tau_off.reshape(T.shape)
    
    sigma_on = spec.calc_pca_T_spectrum(ConstTerms['absPCA']['O2on'],T,ConstTerms['base_T'],ConstTerms['base_P']) # oxygen absorption cross section
    sigma_on = sigma_on.reshape((ConstTerms['absPCA']['O2on']['nu_pca'].size,T.shape[0],T.shape[1]))
    sigma_on = sigma_on.transpose((1,2,0))
    # plt.figure(); plt.plot(nu_pca,sigma_on[0,5,:]);
    
    beta = spec.calc_pca_T_spectrum(ConstTerms['molPCA']['O2'],T,ConstTerms['base_T'],ConstTerms['base_P']) # molecular spectrum
    beta = beta.reshape((ConstTerms['molPCA']['O2']['nu_pca'].size,T.shape[0],T.shape[1]))
    beta = beta.transpose((1,2,0))
    # plt.figure(); plt.plot(nu_pca,beta[0,5,:]);
    
    sigma_off = spec.calc_pca_T_spectrum(ConstTerms['absPCA']['O2off'],T,ConstTerms['base_T'],ConstTerms['base_P']) # oxygen absorption cross section
    sigma_off = sigma_off.reshape(T.shape)
    
    # check gradient function
    sigma_on,dsigma_on = spec.calc_pca_T_spectrum_w_deriv(ConstTerms['absPCA']['O2on'],np.array([280]),ConstTerms['base_T'][0:1],ConstTerms['base_P'][0:1])
    sigma_on_h = spec.calc_pca_T_spectrum(ConstTerms['absPCA']['O2on'],np.array([280.001]),ConstTerms['base_T'][0:1],ConstTerms['base_P'][0:1])
    sigma_on_l = spec.calc_pca_T_spectrum(ConstTerms['absPCA']['O2on'],np.array([279.999]),ConstTerms['base_T'][0:1],ConstTerms['base_P'][0:1])
    plt.figure(); 
    plt.plot((sigma_on_h-sigma_on_l)/0.002,label='Numeric'); 
    plt.plot(dsigma_on,'--',label='Analytic')
    # end Test code
    """
       
    # Channel Order:
    # 'WVOnline', 'WVOffline', 'MolOnline', 'MolOffline', 'CombOnline', 'CombOffline'
    x0 = {}
    x0['xG'] = np.array([-0.1,0.0001,1.6,1.5,1.6,1.6])
    x0['xDT'] = np.array([np.log(30e-9),np.log(30e-9),np.log(30e-9),np.log(30e-9),np.log(30e-9),np.log(30e-9)])
#    x0['xPhi'] = np.log((raw_profs['CombOffline'].profile - raw_profs['CombOffline'].bg[:,np.newaxis])/(ConstTerms['CombOffline']['mult']*np.exp(x0['xG'][5])*profs['Backscatter_Ratio'].profile))
    x0['xPhi'] = np.log((raw_profs['MolOffline'].profile - raw_profs['MolOffline'].bg[:,np.newaxis])/(ConstTerms['MolOffline']['mult']))
    x0['xPsi'] = np.log((raw_profs['WVOffline'].profile - raw_profs['WVOffline'].bg[:,np.newaxis])/ConstTerms['WVOffline']['mult'])
    x0['xN'] = np.log(profs['Absolute_Humidity'].profile*lp.N_A/lp.mH2O)
    x0['xT'] = np.concatenate((np.zeros(profs['Surf_T'].profile.shape),np.diff(profs['Temperature'].profile,axis=1)),axis=1)
    x0['xB'] = np.log(profs['Backscatter_Ratio'].profile-1)
    
    # set nan values to minimum
    for var in x0.keys():
        if np.sum(np.isnan(x0[var])):
            x0[var][np.isnan(x0[var])] = opt_bounds[var][0]
    
    
    initial_profs = td.Build_TD_sparsa_Profiles(x0,ConstTerms,return_params=True,scale=var_scale)     # ,scale={'xB':1,'xN':1,'xT':1,'xPhi':1,'xPsi':1}
    
    ErrorFunc = lambda x: td.TD_sparsa_Error(x,raw_profs,ConstTerms,lam_set,scale=var_scale)
    GradErrorFunc = lambda x: td.TD_sparsa_Error_Gradient(x,raw_profs,ConstTerms,lam_set,scale=var_scale)
    
    
    
    """
    # Test initial profiles
    for var in raw_profs.keys():
        plt.figure();
        plt.semilogy(raw_profs[var].profile[3,:],label='Data'); 
        plt.semilogy(initial_profs[var][3,:],label='Model')
        plt.title(var)
    
    FitError = td.TD_sparsa_Error(x0,raw_profs,ConstTerms,lam_set,scale=var_scale)
    """
    
    
    """
    # Test Gradient Functions
    gradNum = mle.Num_Gradient_Dict(ErrorFunc,x0,step_size=1e-3)
    gradDirect = GradErrorFunc(x0)
    
    for gvar in gradNum.keys():
        plt.figure()
        plt.title(gvar)
        plt.plot(gradNum[gvar].flatten(),label='Numeric')
        plt.plot(gradDirect[gvar].flatten(),'--',label='Direct')
        plt.legend()
    """
    
    sol,[error_hist,step_hist]= mle.GVHSRL_sparsa_optimizor(ErrorFunc,GradErrorFunc,x0,lam_set,sub_eps=1e-5,step_eps=step_eps,opt_cnt_min=10,opt_cnt_max=max_iter,cnt_alpha_max=10,sigma=1e-5,verbose=False,alpha = 1e15)
    sol_profs = td.Build_TD_sparsa_Profiles(sol,ConstTerms,return_params=True,scale=var_scale)

plt.figure()
plt.plot(error_hist)

plt.figure()
plt.plot(step_hist)


#
#        if not verify:
#            prof_sol = mle.Build_WVDIAL_sparsa_Profiles(sol,ConstTerms,dt=rate_adj,return_params=True,scale=scale)
#    
#        
#        ProfileLogError = mle.WVDIAL_sparsa_Error(sol,verify_profs,ConstTerms,lam0,dt=rate_adj,scale=scale)    
#        
#    
#        errRecord.extend([error_hist])
#        stepRecord.extend([step_hist])
#        lam_List.extend([lam.copy()])
#        lam_sets.extend([[lam['xN'],lam['xPhi'],ProfileLogError]])
#        tv_sublist = []
#        for xvar in ['xN','xPhi']:
#            tv_sublist.extend([np.nansum(np.abs(np.diff(sol[xvar],axis=1)))+np.nansum(np.abs(np.diff(sol[xvar],axis=0)))])
#        tv_list.extend([tv_sublist])
#        fitErrors[i_lam] = np.nansum(ProfileLogError)
#        sol_List.extend([sol.copy()])        
#        if verbose:
#            print('Iteration: %d'%i_lam)
#            print('SPARSA iterations: %d'%len(error_hist))
#            print('Log Error: %f'%fitErrors[i_lam])
#            print('lambda nWV, phi')
#            print('    %e | %e '%(lam['xN'],lam['xPhi']))
#            print('TV nWV, phi')
#            print('    %e | %e '%(tv_sublist[0],tv_sublist[1]))
#            print('')
#    
#    ### End Optimization Routine ###
#    
#    #%% Save and Plot Results
#    
#    isol = np.argmin(fitErrors)
#    lam_sets = np.array(lam_sets)
#    tv_list = np.array(tv_list)
#    
#    print('Solution:')
#    print('lambda nWV, phi')
#    print('    %e | %e '%(lam_sets[isol,0],lam_sets[isol,1]))
#    
#    if save_results and lam_sets.shape[0] > 1:
#        np.savez(save_path+save_file+'3D_opt_results_WVDIAL'+datetime.datetime.now().strftime('%Y%m%dT%H%M'),lam_sets=lam_sets,tv_list=tv_list,rate_adj=rate_adj)
#        pickle.dump(ConstTerms,open( save_path+save_file+'_ConstTerms.p', "wb" ))
#        pickle.dump(scale,open( save_path+save_file+'_scale.p', "wb" ))
#        pickle.dump(x0,open( save_path+save_file+'_x0.p', "wb" ))
#        pickle.dump(sol_List[isol],open( save_path+save_file+'_solution.p', "wb" ))
#    
#    lam_error_scale = lam_sets[:,2]-np.nanmin(lam_sets[:,2])
#    lam_error_scale = lam_error_scale/np.nanmax(lam_error_scale)
#
#    if plot_results:
#                
#        # 2D regularizer
#        plt.figure()
#        plt.scatter(lam_sets[:,0],lam_sets[:,1],c=lam_sets[:,2])
#        plt.xlabel(r'$\lambda_{n}$')
#        plt.ylabel(r'$\lambda_{\phi}$')
#        plt.xscale('log')
#        plt.yscale('log')
#        plt.colorbar()
#        if save_results and lam_sets.shape[0] > 1:
#            plt.savefig(save_path+save_file+'_compare_regularizer.png',dpi=300)
#        
#
#        
#        if lam_sets.shape[0] > 1:
#            from matplotlib.mlab import griddata
#            lam_mask = np.ones(lam_error_scale.shape)
#            #lam_mask[lam_error_scale >0.2] = np.nan
#            xgrid = np.logspace(np.log10(lam_sets[:,0].min()),np.log10(lam_sets[:,0].max()),50)
#            ygrid = np.logspace(np.log10(lam_sets[:,1].min()),np.log10(lam_sets[:,1].max()),50)
#            zgrid = griddata(lam_sets[:,0],lam_sets[:,1],lam_error_scale*lam_mask,xgrid,ygrid,interp='linear')
#            plt.figure();
#            #plt.contourf(xgrid,ygrid,zgrid,np.logspace(-5,-1,40));
#            plt.contourf(xgrid,ygrid,zgrid,15);
#            #plt.clim([0,0.1])
#            plt.scatter(lam_sets[:,0],lam_sets[:,1],c=lam_error_scale,lw=1,edgecolor='w')
#            #plt.clim([0,0.1])
#            plt.plot(lam_sets[isol,0],lam_sets[isol,1],'wx')
#            plt.xlabel(r'$\lambda_{n}$')
#            plt.ylabel(r'$\lambda_{\phi}$')
#            plt.xscale('log')
#            plt.yscale('log')
#            #plt.clim([0,0.1])
#            if save_results and lam_sets.shape[0] > 1:
#                plt.savefig(save_path+save_file+'_compare_regularizer_contour.png',dpi=300)
#        
#        plt.figure()
#        plt.plot(errRecord[isol])
#        plt.xlabel('Iterations')
#        plt.ylabel('Log-likelihood')
#        plt.grid(b=True)
#        if save_results:
#            plt.savefig(save_path+save_file+'_LogLikelihood.png',dpi=300)
#        
#        
#        plt.figure()
#        plt.semilogy(stepRecord[isol][1:])
#        plt.xlabel('Iterations')
#        plt.ylabel('Step Evaluation')
#        plt.grid(b=True)
#        if save_results:
#            plt.savefig(save_path+save_file+'_StepEvaluation.png',dpi=300)
#    
#    prof_sol = mle.Build_WVDIAL_sparsa_Profiles(sol_List[isol],ConstTerms,dt=rate_adj,return_params=True,scale=scale)
#    
#    if plot_results:
#        tind = np.int(np.round(np.random.rand()*sol['xN'].shape[0]))
#        subnum = 210
#        plt.figure()
#        for ifig,var in enumerate(fit_profs.keys()):
#            plt.subplot(subnum+1+ifig)
#            plt.semilogy(fit_profs[var].profile[tind,:])
#            plt.semilogy(verify_profs[var].profile[tind,:])
#            plt.semilogy(prof_x0[var][tind,:])
#            plt.semilogy(prof_sol[var][tind,:])  
#        if save_results:
#            plt.savefig(save_path+save_file+'Fitprofs_1D.png',dpi=300)
#        
#        
#        plt.figure()
#        plt.subplot(211)
#        plt.plot(profs['Absolute_Humidity'].profile[tind,:]*lp.N_A/lp.mH2O)
#        plt.plot(prof_x0['nWV'][tind,:])
#        plt.plot(prof_sol['nWV'][tind,:])
#        plt.ylim([0,20*lp.N_A/lp.mH2O])
#        plt.title('$n_{wv}$')
#        plt.subplot(212)
#        plt.semilogy(phi0[tind,:])
#        plt.semilogy(prof_x0['phi'][tind,:])
#        plt.semilogy(prof_sol['phi'][tind,:])
#        plt.title('$\phi$')
#        if save_results:
#            plt.savefig(save_path+save_file+'Abs_Humid_Denoise_1D.png',dpi=300)
#    
#    
#    
#    denoise_labels = ['Absolute_Humidity','Attenuated_Backscatter']
#    denoise_profs = {}
#    
#    var = denoise_labels[0]
#    var1 = 'Offline'
#    
#    denoise_profs[var] = profs[var1].copy()
#    denoise_profs[var].label = 'Denoised '+profs[var].label
#    denoise_profs[var].descript = 'PTV Denoised ' + profs[var].descript
#    denoise_profs[var].profile = prof_sol['nWV']*lp.mH2O/lp.N_A
#    denoise_profs[var].range_array = raw_profs['Offline'].range_array
#    denoise_profs[var].profile_type = profs[var].profile_type
#    denoise_profs[var].profile_variance = np.ones(denoise_profs[var].profile.shape)
#    denoise_profs[var].ProcessingStatus = []
#    denoise_profs[var].cat_ProcessingStatus('PTV denoised')
#    
#    var = denoise_labels[1]
#    var1 = 'Offline'
#    denoise_profs[var] = profs[var1].copy()
#    denoise_profs[var].label = 'Denoised '+ str(var).replace('_',' ')
#    denoise_profs[var].descript = 'PTV Denoised Common terms between online and offline channels'
#    denoise_profs[var].profile = prof_sol['phi']
#    #denoise_profs[var].range_array = raw_profs[var1].range_array
#    denoise_profs[var].profile_variance = np.ones(denoise_profs[var].profile.shape)
#    denoise_profs[var].ProcessingStatus = []
#    denoise_profs[var].cat_ProcessingStatus('PTV denoised')
#    
#    
#      
#    
#    if plot_results:          
#        for plt_var in denoise_profs.keys():
#            lplt.pcolor_profiles([denoise_profs[plt_var]],cmap=[plot_settings[plt_var]['colormap']],scale=[plot_settings[plt_var]['scale']],
#                                 climits=[plot_settings[plt_var]['climits']],plot_date=True)
#            if save_results:
#                plt.savefig(save_path+save_file+'_'+str(plt_var)+'_Denoise_'+plot_settings[plt_var]['colormap']+'.png',dpi=600)
#           
#        prof_vars = ['Offline','Aerosol_Backscatter_Coefficient','Absolute_Humidity']     
#        for plt_var in prof_vars:
#            if plt_var in profs:
#                lplt.pcolor_profiles([profs[plt_var]],cmap=[plot_settings[plt_var]['colormap']],scale=[plot_settings[plt_var]['scale']],
#                                     climits=[plot_settings[plt_var]['climits']],plot_date=True)
#                if save_results:
#                    plt.savefig(save_path+save_file+'_'+str(plt_var)+'_'+plot_settings[plt_var]['colormap']+'.png',dpi=600)
#    
#    
#    if save_results:
#        for var in denoise_profs.keys():
#            denoise_profs[var].write2nc(save_path+save_file+'.nc')
#        for var in profs.keys():
#            profs[var].write2nc(save_path+save_file+'.nc')
#        for var in fit_profs.keys():
#            fit_profs[var].write2nc(save_path+save_file+'.nc')
#        for var in verify_profs.keys():
#            verify_profs[var].write2nc(save_path+save_file+'.nc')
#        lp.write_var2nc(lam_sets,'regularizer',save_path+save_file+'.nc')
#        lp.write_var2nc(step_eps,'step_eps',save_path+save_file+'.nc')
#        lp.write_var2nc(max_iter,'max_iter',save_path+save_file+'.nc')
#    