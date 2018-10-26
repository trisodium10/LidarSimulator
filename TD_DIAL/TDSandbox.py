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
lam_range0 = {'xB':[0,2],'xN':[1.0,2.0],'xT':[2.0,3.0],'xPhi':[1.0,2.0],'xPsi':[1.0,2.0]}  # note these search bounds are log10
#Num_reg_iter = 1
verbose = False
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
    LidarNumber = np.int(profs[var].lidar[-1])

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
    x0['xG'] = [0,0,-1,0,-1,0]
    x0['xDT'] = [np.log(30e-9),np.log(30e-9),np.log(30e-9),np.log(30e-9),np.log(30e-9),np.log(30e-9)]
    x0['xPhi'] = np.log((raw_profs['CombOffline'].profile - raw_profs['CombOffline'].bg[:,np.newaxis])/ConstTerms['CombOffline']['mult'])
    x0['xPsi'] = np.log((raw_profs['WVOffline'].profile - raw_profs['WVOffline'].bg[:,np.newaxis])/ConstTerms['WVOffline']['mult'])
    x0['xN'] = np.log(profs['Absolute_Humidity'].profile*lp.N_A/lp.mH2O)
    x0['xT'] = np.concatenate((profs['Surf_T'].profile,np.diff(profs['Temperature'].profile,axis=1)),axis=1)
    x0['xB'] = np.log(profs['Backscatter_Ratio'].profile-1)
    
    # set nan values to minimum
    for var in x0.keys():
        if np.sum(np.isnan(x0[var])):
            x0[var][np.isnan(x0[var])] = opt_bounds[var][0]
    
    
    initial_profs = td.Build_TD_sparsa_Profiles(x0,ConstTerms,dR=37.5,return_params=True,scale=var_scale)     # ,scale={'xB':1,'xN':1,'xT':1,'xPhi':1,'xPsi':1}
    
    FitError = td.TD_sparsa_Error(x0,raw_profs,ConstTerms,lam_set,scale=var_scale)
    
    """
    # Test initial profiles
    var = 'CombOnline'
    plt.figure();
    plt.semilogy(raw_profs[var].profile[3,:]); 
    plt.semilogy(initial_profs[var][3,:])
    """