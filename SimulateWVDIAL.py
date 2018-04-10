# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:20:56 2017

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

import TDRetrievalLib as td

import datetime

use_RD = True  # simulate Rayleigh-Doppler Effect
use_shot_noise = False # simulate shot noise

Nprofiles = 5*30  # number of two second profiles

wavelength_list = [828.3026e-9,828.203e-9,780.246119e-9,780.246119e-9,769.2339e-9,769.319768e-9]  # offline,online,Comb,Mol,online,offline  # O2 On center: 769.2339e-9
K_list = Nprofiles*np.array([0.5e16,0.5e16,1.27e16*0.3,1.15e16*0.7,0.5e16,0.5e16])  # list of multiplier constants
bg_list = Nprofiles*np.array([1.0,1.0,0.64,0.29,1.0,1.0])*5.7  # list of background levels (6 night, 150 peak day clear, 700-1000 day cloudy)
name_list = ['WV Offline','WV Online','HSRL Combined','HSRL Molecular','O2 Online','O2 Offline']  # name corresponding to each wavelength

signal_list = [];

sim_dr = 37.5/8  # simulated range resolution
sim_LR = 35.0 # simulated lidar ratio

sim_dnu = 10e6  # spacing in optical frequency space of simulation
sim_nu_max = 5e9  # edge of simulated frequency space

sim_T0 = 300  # temp at surface in K
sim_dT1 = 6.5e-3  # lapse rate in K/m
sim_dT2 = 9.3e-3  # lapse rate in K/m
zT = 2.6e3        # transition between lapse rates

PL_laser = 150  # resolution due to laser pulse length in m

bin_res = 37.5  # range bin resolution

TauP = 8e3  # pressure decay constant in m altitude

# fraction of O2 by number density
fO2 = (32*0.2320+28.02*0.7547+44.01*0.00046+39.94*0.0128+20.18*0.000012+4.0*0.0000007+83.8*0.000003+131.29*0.00004)*0.2320/32.0

betaScat0 = 0 #5e-11  # "scattering coefficient" of scattering by transmit pulse
TauScat0 = 5  # decay constant in m (range) of scattering in the instrument from transmit pulse

beta_c = 1e-4 # 4e-4   # cloud peak bs coefficient
cloud_wid = 100  # could width in m
cloud_alt = 6000 # cloud centroid altitude in m
sim_cloud_LR = 18  # cloud lidar ratio

#geo_file = '/h/eol/mhayman/PythonScripts/NCAR-LidarProcessing/calibrations/geo_DLB_20170512.npz'
geo_file = '/Users/mhayman/Documents/Python/Lidar/NCAR-LidarProcessing/calibrations/geo_DLB_20170512.npz'

o2_spec_file = '/Users/mhayman/Documents/DIAL/O2_HITRAN2012_760_781.txt'

sim_range = np.arange(0,15e3,sim_dr)  # range array in m
sim_nu = np.arange(-sim_nu_max,sim_nu_max,sim_dnu) # frequency array in Hz
inu0 = np.argmin(np.abs(sim_nu)) # index to center of frequency array

# aerosol backscatter coefficient m^-1 sr^-1 at 780 nm
beta_aer_hsrl = 1e-8*np.ones(sim_range.size)
beta_aer_hsrl[np.nonzero(sim_range < 2e3)] = 1e-6
p_sim_aer = np.polyfit(sim_range,np.log10(beta_aer_hsrl),13)
sim_beta_aer = 10**(np.polyval(p_sim_aer,sim_range))
#sim_beta_aer = sim_beta_aer + np.exp(-(sim_range-cloud_alt)**2/(cloud_wid**2))

sim_beta_cloud = beta_c*np.exp(-(sim_range-cloud_alt)**2/(cloud_wid**2))

## aerosol extinction coefficient at 780 nm
#alpha_aer_hsrl = beta_aer_hsrl*sim_LR

# absolute humidity initally in g/m^3
sim_nWVi = 20*(1-sim_range/6e3)
sim_nWVi[np.nonzero(sim_range > 4e3)] = 0.6
p_sim_wv = np.polyfit(sim_range,np.log(sim_nWVi),13)
sim_nWV = np.exp(np.polyval(p_sim_wv,sim_range))
sim_nWV = sim_nWV*lp.N_A/lp.mH2O  # convert to number/m^3

# create a piecewise temperature profile
# use polyfit to add a few wiggles
i_zT = np.nonzero(sim_range > zT)
sim_T_base = sim_T0-sim_dT1*sim_range
sim_T_base[i_zT] =(sim_T0-sim_dT1*zT+sim_dT2*zT)-sim_dT2*sim_range[i_zT]
p_sim_T = np.polyfit(sim_range,sim_T_base,3)
sim_T = np.polyval(p_sim_T,sim_range)

#plt.figure();
#plt.plot(sim_T_base,sim_range)
#plt.plot(sim_T,sim_range)

# pressure profile in atm.
sim_P = 1.0*np.exp(-sim_range/TauP)

sim_nO2 = fO2*(sim_P*101325/(lp.kB*sim_T)-sim_nWV)

# Overlap function definition
itop = 75
ol_sim_range = np.arange(sim_range[-1]/100.0+1)*100
geo_data = np.load(geo_file)
geo_corr = geo_data['geo_prof']
geo_corr0 = geo_corr[itop,1]  # normalize to bin 100
geo_corr[:,1] = geo_corr[:,1]/geo_corr0
itop_sim = np.argmin(np.abs(ol_sim_range-geo_corr[itop,0]))
pgeo = np.polyfit(geo_corr[1:itop,0],np.log(geo_corr[1:itop,1]),8)
sim_geo = np.polyval(pgeo,ol_sim_range)
sim_geo[itop_sim:] = 0.0
sim_geo[np.nonzero(ol_sim_range < geo_corr[1,0])] = np.max(sim_geo)
sim_geo_grid = np.interp(sim_range,ol_sim_range,sim_geo,left=np.max(sim_geo),right=0.0)
Overlap = 1.0/np.exp(sim_geo_grid)

#plt.figure(); 
#plt.plot(geo_corr[:,0],1.0/geo_corr[:,1])
#plt.plot(sim_range,Overlap)

# Simulate scattering in the receiver as an exponential decay
tx_scatter = np.exp(-sim_range/TauScat0)
tx_scatter = betaScat0*tx_scatter/np.sum(tx_scatter)

# etalon design
Etalon_angle = 0.00*np.pi/180
Filter = FO.FP_Etalon(1.0932e9,43.5e9,lp.c/wavelength_list[1],efficiency=0.95,InWavelength=False)

HSRL_Etalon_angle = 0.00*np.pi/180
HSRL_Filter1 = FO.FP_Etalon(5.7e9,45e9,lp.c/wavelength_list[2],efficiency=0.95,InWavelength=False)
HSRL_Filter2 = FO.FP_Etalon(15e9,250e9,lp.c/wavelength_list[2],efficiency=0.95,InWavelength=False)

O2_Etalon_angle = 0.00*np.pi/180
O2_Filter = FO.FP_Etalon(1.0932e9,43.5e9,lp.c/wavelength_list[4],efficiency=0.95,InWavelength=False)

sim_pulse = np.ones(np.round(PL_laser/sim_dr).astype(np.int))
sim_pulse = sim_pulse/sim_pulse.size

bin_func = np.ones(np.round(bin_res/sim_dr).astype(np.int))
sim_range_bin = np.arange(sim_range[-1]/bin_res)*bin_res


# Assume 97% Rb 87
Lcell = 7.2e-2  # Rb Cell length in m
Tcell = 50+274.15 # Rb Cell Temperature in K
K85, K87 = spec.RubidiumD2Spectra(Tcell,sim_nu+lp.c/wavelength_list[2],0.0)  # Calculate absorption coefficeints
RbFilter = np.exp(-Lcell*(np.sum(K87,axis=0)*0.97/0.27832+np.sum(K85,axis=0)*0.03/0.72172))  # calculate transmission accounting for isotopic fraction of the cell


# Oxygen Spectroscopy
ext_o2 = spec.ExtinctionFromHITRAN(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[4],sim_T[0:1],sim_P[0:1],(td.mO2*1e-3)/lp.N_A,nuLim=np.array([lp.c/770e-9,lp.c/768e-9]),freqnorm=True,filename=o2_spec_file).T
Tetalon = O2_Filter.spectrum(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[4],InWavelength=False,aoi=Etalon_angle,transmit=True)
plt.figure(); 
plt.stem(1e9*np.array([wavelength_list[4],wavelength_list[5]]),np.ones(2),label='Laser')
plt.plot(1e9*lp.c/(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[4]),Tetalon,label='Etalon'); 
plt.plot(1e9*lp.c/(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[4]),ext_o2[:,0]/np.max(ext_o2[:,0]),label='Oxygen'); 
plt.grid(b=True); 
plt.xlabel('Wavelength [nm]'); 
plt.ylabel('Transmission [A.U.]');
plt.legend();

# WV Spectroscopy
#ext_wv = lp.WV_ExtinctionFromHITRAN(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[1],sim_T[0:1],sim_P[0:1],nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9]),freqnorm=True).T
#Tetalon = Filter.spectrum(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[1],InWavelength=False,aoi=Etalon_angle,transmit=True)
#plt.figure(); 
#plt.plot(1e9*lp.c/(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[1]),Tetalon); 
#plt.plot(1e9*lp.c/(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[1]),ext_wv[:,0]/np.max(ext_wv[:,0])); 
#plt.grid(b=True); 
#plt.xlabel('Wavelength [nm]'); 
#plt.ylabel('Transmission [A.U.]');
#plt.legend(['Etalon','Water Vapor']);

sim_beta_mol = 5.45*(550.0e-9/wavelength_list[2])**4*1e-32*(sim_P/9.86923e-6)/(sim_T*lp.kB)

#ilist = 0
for ilist in range(len(wavelength_list)):

    """
    use convention [frequency,range] in 2D arrays
    """
    beta_mol = 5.45*(550.0e-9/wavelength_list[ilist])**4*1e-32*(sim_P/9.86923e-6)/(sim_T*lp.kB)
    beta_aer = sim_beta_aer*wavelength_list[ilist]/780.24e-9
    beta_cloud = sim_beta_cloud  #*wavelength_list[ilist]/780.24e-9  # No wavelength dependence in cloud
    alpha_aer = beta_aer*sim_LR + beta_cloud*sim_cloud_LR
    OD_aer = np.cumsum(alpha_aer)*sim_dr
    OD_mol = np.cumsum(beta_mol*8*np.pi/3)*sim_dr
    
    Taer = np.exp(-OD_aer-OD_mol)
    
    BSR = (beta_mol+beta_aer+beta_cloud)/beta_mol
    
    
    
    # obtain the molecular backscatter spectrum
    molBeta_nu = lp.RB_Spectrum(sim_T,sim_P,wavelength_list[ilist],nu=sim_nu,norm=True)
    #plt.figure(); 
    #plt.imshow(molBeta_nu);
    
    if 'WV' in name_list[ilist]:
        # define etalon transmission
        Tetalon = Filter.spectrum(sim_nu+lp.c/wavelength_list[ilist],InWavelength=False,aoi=Etalon_angle,transmit=True)        
        
        # obtain frequency resolved water vapor extinction coefficient
        ext_wv = lp.WV_ExtinctionFromHITRAN(lp.c/wavelength_list[ilist]+sim_nu,sim_T,sim_P,nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9]),freqnorm=True).T
            
        
        OD_wv = np.cumsum(sim_nWV[np.newaxis,:]*ext_wv,axis=1)*sim_dr  # obtain frequency resolved optical depth
        
        T_tx = np.exp(-OD_wv[inu0,:])  # outgoing atmospheric transmission
        T_rx_aer = np.exp(-OD_wv[inu0,:])*Tetalon[inu0]  # return transmission for atmosphere and etalon seen by aerosols
        if use_RD:
            T_rx_mol = np.sum(molBeta_nu*Tetalon[:,np.newaxis]*np.exp(-OD_wv),axis=0)          
        else:
            T_rx_mol = np.exp(-OD_wv[inu0,:])*Tetalon[inu0]    
    elif 'O2' in name_list[ilist]:
        # define etalon transmission
        Tetalon = O2_Filter.spectrum(sim_nu+lp.c/wavelength_list[ilist],InWavelength=False,aoi=Etalon_angle,transmit=True)          
        
#        ext_o2 = lp.WV_ExtinctionFromHITRAN(lp.c/wavelength_list[ilist]+sim_nu,sim_T,sim_P,nuLim=np.array([lp.c/770e-9,lp.c/768e-9]),freqnorm=True,filename=o2_spec_file).T
        ext_o2 = spec.ExtinctionFromHITRAN(lp.c/wavelength_list[ilist]+sim_nu,sim_T,sim_P,(td.mO2*1e-3)/lp.N_A,nuLim=np.array([lp.c/770e-9,lp.c/768e-9]),freqnorm=True,filename=o2_spec_file).T
        
        OD_o2 = np.cumsum(sim_nO2[np.newaxis,:]*ext_o2,axis=1)*sim_dr
        
        T_tx = np.exp(-OD_o2[inu0,:])
        T_rx_aer = np.exp(-OD_o2[inu0,:])*Tetalon[inu0]
        if use_RD:
            T_rx_mol = np.sum(molBeta_nu*Tetalon[:,np.newaxis]*np.exp(-OD_o2),axis=0)
        else:
            T_rx_mol = np.exp(-OD_o2[inu0,:])*Tetalon[inu0]
        
    else:
        Tetalon = HSRL_Filter1.spectrum(sim_nu+lp.c/wavelength_list[2],InWavelength=False,aoi=Etalon_angle,transmit=True) \
            *HSRL_Filter2.spectrum(sim_nu+lp.c/wavelength_list[2],InWavelength=False,aoi=Etalon_angle,transmit=True)
        BSR_780 = BSR  # store for comparison to processing retrievals
        if 'Molecular' in name_list[ilist]:
            T_rx_mol = np.sum(molBeta_nu*Tetalon[:,np.newaxis]*RbFilter[:,np.newaxis],axis=0)
            T_rx_aer = Tetalon[inu0]*RbFilter[inu0]
            T_tx = 1.0
        else:
            T_rx_mol = np.sum(molBeta_nu*Tetalon[:,np.newaxis],axis=0)
            T_rx_aer = Tetalon[inu0]
            T_tx = 1.0
    
#    plt.figure()
#    plt.plot(T_wv_tx,sim_range)
#    plt.plot(T_wv_rx_mol,sim_range)
    
    
    BS_Sig = Overlap*Taer**2*T_tx*(T_rx_mol*beta_mol+T_rx_aer*(beta_aer+beta_cloud))/sim_range**2
    BS_Sig[np.nonzero(sim_range==0)] = 0
    BS_Sig = BS_Sig+tx_scatter*Overlap[0]
    
    # Add constant multipliers
    BS_Sig = BS_Sig*K_list[ilist]*sim_dr/37.5
    
    #Add background
    BS_Sig = BS_Sig+bg_list[ilist]*sim_dr/37.5
    
#    # pulse convolution    
    BG_Sig = np.convolve(BS_Sig,sim_pulse,mode='same')
    
    # resample at lidar range resolution
    BS_Sig = np.convolve(BS_Sig,bin_func,mode='same')
    BS_Sig = np.interp(sim_range_bin,sim_range,BS_Sig)
    
    # Nonlinearity?

    # Shot Noise 
    if use_shot_noise:   
        BS_Sig = np.random.poisson(lam=BS_Sig)

    Prof = lp.LidarProfile(BS_Sig[np.newaxis,:],np.array([0]),label='Simulated '+name_list[ilist],lidar='WV-DIAL',binwidth=bin_res*2/lp.c,wavelength=wavelength_list[ilist],StartDate=datetime.datetime.today())
    signal_list.extend([Prof])    
    
#    if ilist == 0:
#        plt.figure()
#    plt.semilogx(BS_Sig,sim_range)


#### compute full line absorption
#nu_line = np.linspace(lp.c/wavelength_list[0]-sim_nu_max,lp.c/wavelength_list[1]+sim_nu_max,10000)
#ext_wv_off = lp.WV_ExtinctionFromHITRAN(nu_line,sim_T,sim_P,nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9]),freqnorm=True)
#Tetalon = Filter.spectrum(nu_line,InWavelength=False,aoi=Etalon_angle,transmit=True)
#plt.figure();
#plt.plot(1e-9*(nu_line-lp.c/wavelength_list[0]),ext_wv_off[0,:]); 
#plt.plot(1e-9*sim_nu,molBeta_nu[:,0]/np.max(molBeta_nu[:,0])*np.max(ext_wv_off[0,:]))
#plt.plot(1e-9*(sim_nu+lp.c/wavelength_list[1]-lp.c/wavelength_list[0]),molBeta_nu[:,0]/np.max(molBeta_nu[:,0])*np.max(ext_wv_off[0,:]))
#plt.plot(1e-9*(nu_line-lp.c/wavelength_list[0]),Tetalon*np.max(ext_wv_off[0,:]));
#plt.legend(['WV Abs.','Off-line RB','Online RB','Etalon'],loc=9)
#plt.grid(b=True)
#plt.xlabel('Frequency Offset from Offline [GHz]')
#### end compute full line absorption







"""
Process Data (Standard routines)
"""

lp.plotprofiles(signal_list)

for ai, signal in enumerate(signal_list):
    signal.bg_subtract(-50)

lp.plotprofiles(signal_list)

beta_mol_sonde,temp,pres = lp.get_beta_m_model(signal_list[2],np.array([sim_T[0]]),np.array([sim_P[0]]),returnTP=True)
pres.gain_scale(9.86923e-6)  
pres.profile_type = '$atm.$'

## raw signal convolution
#signal_list[0].conv(0.0,1.0/bin_res)
#signal_list[1].conv(0.0,1.0/bin_res)


nWV = wv.WaterVapor_Simple(signal_list[1],signal_list[0],pres.profile.flatten(),temp.profile.flatten())
nWV.conv(0.0,np.sqrt(150**2+75**2)/bin_res)


RbFilterProc = np.exp(-Lcell*(np.sum(K87,axis=0)*0.97/0.27832+np.sum(K85,axis=0)*0.03/0.72172))
TetalonProc = HSRL_Filter1.spectrum(sim_nu+lp.c/wavelength_list[2],InWavelength=False,aoi=Etalon_angle,transmit=True) \
            *HSRL_Filter2.spectrum(sim_nu+lp.c/wavelength_list[2],InWavelength=False,aoi=Etalon_angle,transmit=True)           
molBeta_nu_Proc = lp.RB_Spectrum(temp.profile.flatten(),pres.profile.flatten(),wavelength_list[2],nu=sim_nu,norm=True)

eta_mol = np.sum(molBeta_nu_Proc*TetalonProc[:,np.newaxis]*RbFilterProc[:,np.newaxis],axis=0)
        
# use scalar for molecular gain
#molGainScale = 1.50  # scale for molecular channel
#signal_list[3].gain_scale(molGainScale)

# range dependent molecular gain adjustment
#plt.figure(); 
#plt.plot(signal_list[2].profile.flatten()/signal_list[3].profile.flatten())
#signal_list[3].diff_geo_overlap_correct(0.9855*3/7.0/eta_mol)
mol = signal_list[3].copy()
mol.gain_scale(1.69)
comb = signal_list[2]
aer_beta = lp.AerosolBackscatter(mol,comb,beta_mol_sonde)

bsr = aer_beta.copy()
bsr.profile = aer_beta.profile+beta_mol_sonde.profile
bsr.divide_prof(beta_mol_sonde)
bsr.label = 'Backscatter Ratio'
bsr.descript = 'Backscatter Ratio at 780 nm'
bsr.profile_type = ''


# direct calculation of o2 extinction coefficient
alpha_o2 = nWV.copy()
alpha_o2.profile = -1.0/(2)*np.diff(np.log(signal_list[4].profile/signal_list[5].profile),axis=1)/signal_list[5].mean_dR
alpha_o2.label = 'Oxygen Differential Extinction'
alpha_o2.descript = 'Oxygen Differential Extinction'
alpha_o2.profile_type = '$m^{-1}$'
#nWV.range_array = range_diff
alpha_o2.profile_variance = (0.5/signal_list[4].mean_dR)**2*( \
    signal_list[4].profile_variance[:,1:]/signal_list[4].profile[:,1:]**2 + \
    signal_list[4].profile_variance[:,:-1]/signal_list[4].profile[:,:-1]**2 + \
    signal_list[5].profile_variance[:,1:]/signal_list[5].profile[:,1:]**2 + \
    signal_list[5].profile_variance[:,:-1]/signal_list[5].profile[:,:-1]**2)



plt.figure()
plt.plot(sim_nWV/lp.N_A*lp.mH2O,sim_range*1e-3,'k--')
plt.plot(nWV.profile.flatten(),nWV.range_array*1e-3,'r')
plt.xlim([0,30])
plt.xlabel('Absolute Humidity [$g/m^3$]')
plt.ylabel('Altitude [km]')
plt.grid(b=True)

plt.figure()
plt.plot(sim_T,sim_range*1e-3,'k--')
plt.plot(temp.profile.flatten(),temp.range_array*1e-3,'b')
#plt.xlim([150,300])
plt.xlabel('Temperature [$K$]')
plt.ylabel('Altitude [km]')
plt.grid(b=True)

plt.figure()
plt.semilogx(sim_beta_aer+sim_beta_cloud,sim_range*1e-3,'k--')
plt.plot(aer_beta.profile.flatten(),aer_beta.range_array*1e-3,'r')
#plt.xlim([150,300])
plt.xlabel('Particle Backscatter Coefficient [$m^{-1}sr^{-1}$]')
plt.ylabel('Altitude [km]')
plt.grid(b=True)

plt.figure(); 
plt.plot(alpha_o2.profile.flatten(),alpha_o2.range_array*1e-3);
plt.xlim([0,6e-4])
plt.xlabel('Oxygen Extinction [$m^{-1}$]')
plt.ylabel('Altitude [km]')
plt.grid(b=True)


lp.LidarProfile(BS_Sig[np.newaxis,:],np.array([0]),label='Simulated '+name_list[ilist],lidar='WV-DIAL',binwidth=bin_res*2/lp.c,wavelength=wavelength_list[ilist],StartDate=datetime.datetime.today())

range_limits = [200,6e3] # [500,6e3]

nWVAct = nWV.copy()
nWVAct.profile = np.interp(nWVAct.range_array,sim_range,sim_nWV)[np.newaxis,:]
nWVAct.slice_range(range_lim=range_limits)
nWVAct.label = 'Absolute Humidity'
nWVAct.descript = 'Actual Absolute Humidity'
nWVAct.profile_type = '$m^{-3}$'

TAct = nWV.copy()
TAct.profile = np.interp(TAct.range_array,sim_range,sim_T)[np.newaxis,:]
TAct.slice_range(range_lim=range_limits)
TAct.label = 'Temperature'
TAct.descript = 'Actual Temperature'
TAct.profile_type = 'K'

BSRAct = nWV.copy()
BSRAct.profile = np.interp(BSRAct.range_array,sim_range,BSR_780)[np.newaxis,:]
BSRAct.slice_range(range_lim=range_limits)
BSRAct.label = 'Backscatter Ratio'
BSRAct.descript = 'Actual Backscatter Ratio at 780 nm'
BSRAct.profile_type = ''


# compute signal ratios for optimization routines
Hratio = signal_list[2].copy()
Hratio.divide_prof(signal_list[3])
Hratio.slice_range(range_lim=range_limits)

Wratio = signal_list[1].copy()
Wratio.divide_prof(signal_list[0])
Wratio.slice_range(range_lim=range_limits)

Oratio = signal_list[4].copy()
Oratio.divide_prof(signal_list[5])
Oratio.slice_range(range_lim=range_limits)

presfit = pres.copy()
#presfit.gain_scale(101325)  
#presfit.profile_type = '$Pa$'
presfit.slice_range(range_lim=range_limits)

tempfit = temp.copy()
tempfit.slice_range(range_lim=range_limits)

nWVfit = nWV.copy()
nWVfit.gain_scale(lp.N_A/lp.mH2O)
nWVfit.slice_range(range_lim=range_limits)

BSRfit = bsr.copy()
BSRfit.slice_range(range_lim=range_limits)
#
#x02D = np.zeros((Oratio.range_array.size,3))
#x02D[:,0] = np.interp(Oratio.range_array,temp.range_array,temp.profile.flatten())
#x02D[:,1] = np.interp(Wratio.range_array,nWV.range_array,nWV.profile.flatten()*lp.N_A/lp.mH2O)
#x02D[:,2] = np.interp(Hratio.range_array,bsr.range_array,bsr.profile.flatten())
#
#ilim = x02D.shape[0]
#x0 = x02D.T.flatten()
#
#bnds = np.zeros((x0.size,2))
#bnds[:2*ilim,0] = 0
#bnds[2*ilim:,0] = 1.0
#bnds[:ilim,1] = 400.0
#bnds[ilim:2*ilim,1] = 1e25
#bnds[2*ilim:,1] = 1e5
#
#Herror = lambda x: (td.HSRLProfileRatio(x[:ilim],presfit.profile.flatten(),x[2*ilim:],Hratio.wavelength,HSRL_Filter1,GainRatio=0.9855*3/7.0)-Hratio.profile.flatten())**2/Hratio.profile_variance.flatten()
#Werror = lambda x: (td.WaterVaporProfileRatio(x[:ilim],presfit.profile.flatten(),x[ilim:2*ilim],x[2*ilim:],signal_list[1].wavelength,signal_list[0].wavelength,Filter,Wratio.mean_dR,GainRatio=1.0) \
#    -Wratio.profile.flatten())**2/Wratio.profile_variance.flatten()
#Oerror = lambda x: (td.OxygenProfileRatio(x[:ilim],presfit.profile.flatten(),x[ilim:2*ilim],x[2*ilim:],signal_list[4].wavelength,signal_list[5].wavelength,O2_Filter,Oratio.mean_dR,GainRatio=1.0) \
#    -Oratio.profile.flatten())**2/Oratio.profile_variance.flatten()
#    
#ProfError = lambda x: np.nansum(Herror(x)+Werror(x)+Oerror(x))
#
#
#
#
##x0 = np.zeros((Oratio.range_array.size+1,3))
##x02D = np.zeros((Oratio.range_array.size,3))
##x02D[1:,0] = np.interp(Oratio.range_array,temp.range_array,temp.profile.flatten())
##x02D[1:,1] = np.interp(Wratio.range_array,nWV.range_array,nWV.profile.flatten()*lp.N_A/lp.mH2O)
##x02D[1:,2] = np.interp(Hratio.range_array,bsr.range_array,bsr.profile.flatten())
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
##bnds[0,0] = 0
##bnds[0,1] = 10
##bnds[ilim,0] = 0
##bnds[ilim,1] = 10
##bnds[2*ilim,0] = 0
##bnds[2*ilim,1] = 10
##
##Herror = lambda x: (td.HSRLProfileRatio(x[1:ilim],presfit.profile.flatten(),x[2*ilim+1:],Hratio.wavelength,HSRL_Filter1,GainRatio=x[2*ilim])-Hratio.profile.flatten())**2/Hratio.profile_variance.flatten()
##Werror = lambda x: (td.WaterVaporProfileRatio(x[1:ilim],presfit.profile.flatten(),x[ilim+1:2*ilim],x[2*ilim+1:],signal_list[1].wavelength,signal_list[0].wavelength,Filter,Wratio.mean_dR,GainRatio=x[ilim]) \
##    -Wratio.profile.flatten())**2/Wratio.profile_variance.flatten()
##Oerror = lambda x: (td.OxygenProfileRatio(x[1:ilim],presfit.profile.flatten(),x[ilim+1:2*ilim],x[2*ilim+1:],signal_list[4].wavelength,signal_list[5].wavelength,O2_Filter,Oratio.mean_dR,GainRatio=x[0]) \
##    -Oratio.profile.flatten())**2/Oratio.profile_variance.flatten()
##    
##ProfError = lambda x: Herror(x)+Werror(x)+Oerror(x)
#
#sol1D,opt_iterations,opt_exit_mode = scipy.optimize.fmin_tnc(ProfError,x0,bounds=bnds) #,maxfun=2000,eta=1e-5,disp=0)  
##sol2D = np.reshape(sol1D,x02D.shape)


"""
Analysis definitions
"""
a_wavelength_list = [wavelength_list[0],wavelength_list[1],wavelength_list[2],wavelength_list[4],wavelength_list[5]]
#a_wavelength_list = [828.3026e-9,828.203e-9,780.246119e-9,769.2365e-9+0.001e-9,769.319768e-9]  # offline,online,Comb,Mol,online,offline  # O2 On center: 769.2339e-9
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
rb_spec = dict(zip(a_name_list,[wv_rb_off,wv_rb_on,hsrl_rb,o2_rb_on,o2_rb_off]))
abs_spec = dict(zip(a_name_list[0:2]+a_name_list[3:5],[wv_abs_off,wv_abs_on,o2_abs_on,o2_abs_off]))

# define etalon objects for each channel
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

Prof = {}
ProfVar = {}
Prof['HSRL'] = Hratio.profile.flatten()
ProfVar['HSRL'] = Hratio.profile_variance.flatten()

Prof['WV'] = Wratio.profile.flatten()
ProfVar['WV'] = Wratio.profile_variance.flatten()

Prof['O2'] = Oratio.profile.flatten()
ProfVar['O2'] = Oratio.profile_variance.flatten()

# index into laser frequency
inuL = {}
inuL['HSRL'] = inu0
inuL['WV Online'] = inu0
inuL['WV Offline'] = inu0
inuL['O2 Online'] = inu0
inuL['O2 Offline'] = inu0

xAct = np.zeros((Hratio.range_array.size+1,3))
xAct[0,0] = K_list[2]/K_list[3]
xAct[0,1] = 1.0
xAct[0,2] = 1.0
xAct[1:,0] = TAct.profile.flatten()
xAct[1:,1] = nWVAct.profile.flatten()
xAct[1:,2] = BSRAct.profile.flatten()

x0 = np.zeros((Hratio.range_array.size+1,3))
x0[0,0] = 2.0 #1.69
x0[0,1] = 1.0
x0[0,2] = 1.0
x0[1:,0] = tempfit.profile.flatten()
x0[1:,1] = nWVfit.profile.flatten()
x0[1:,2] = BSRfit.profile.flatten()

bndL = np.zeros(x0.shape)
bndU = np.zeros(x0.shape)
bndL[0,0] = x0[0,0]*0.8
bndU[0,0] = x0[0,0]*1.2
bndL[0,1] = x0[0,1]*0.8
bndU[0,1] = x0[0,1]*1.2
bndL[0,2] = x0[0,2]*0.8
bndU[0,2] = x0[0,1]*1.2

bndL[1:,0] = 180
bndU[1:,0] = 310
bndL[1:,1] = 0
bndU[1:,1] = 1e24
bndL[1:,2] = 1
bndU[1:,2] = 1e4

bnds = np.hstack((bndL.flatten()[:,np.newaxis],bndU.flatten()[:,np.newaxis]))

x0 = x0.flatten()

# set out of bounds values to bound limits
iL = np.nonzero(x0<bndL.flatten())[0]
x0[iL] = bnds[iL,0]
iU = np.nonzero(x0<bndU.flatten())[0]
x0[iU] = bnds[iU,1]

Efit = lambda x: td.TDErrorFunction(Prof,ProfVar,x,presfit.profile.flatten(),Trx,rb_spec,abs_spec,Hratio.mean_dR,inuL,multBSR)
gradE = lambda x: td.TDGradientFunction(Prof,ProfVar,x0,presfit.profile.flatten(),Trx,rb_spec,abs_spec,Hratio.mean_dR,inuL,multBSR)
#gradEnum = td.Num_Gradient(Efit,x0,step_size=1e-3)


# Check ratio functions:
fit_test =td.TDRatios(Prof,xAct,presfit.profile.flatten(),Trx,rb_spec,abs_spec,Hratio.mean_dR,inuL,multBSR)
plt.figure()
plt.semilogx(fit_test[0],Hratio.range_array,label='Fit Estimate')
plt.semilogx(Hratio.profile.flatten(),Hratio.range_array,label='Signal Ratio')
plt.semilogx(BSRAct.profile.flatten(),BSRAct.range_array,'k--',label='Actual BSR')
plt.title('HSRL Ratio')
plt.grid(b=True)
plt.legend()

plt.figure()
plt.semilogx(fit_test[1],Wratio.range_array,label='Fit Estimate')
plt.semilogx(Wratio.profile.flatten(),Wratio.range_array,label='Signal Ratio')
plt.title('Water Vapor Ratio')
plt.grid(b=True)
plt.legend()

plt.figure()
plt.semilogx(1./fit_test[2],Oratio.range_array,label='Fit Estimate')
plt.semilogx(Oratio.profile.flatten(),Oratio.range_array,label='Signal Ratio')
plt.title('Oxygen Ratio')
plt.grid(b=True)
plt.legend()

"""

sol1D,opt_iterations,opt_exit_mode = scipy.optimize.fmin_tnc(Efit,x0,fprime=gradE,bounds=bnds) #,bounds=bnds) #,maxfun=2000,eta=1e-5,disp=0)  
sol2D = sol1D.reshape((Hratio.range_array.size+1,3))

fits = td.TDRatios(Prof,sol1D,presfit.profile.flatten(),Trx,rb_spec,abs_spec,Hratio.mean_dR,inuL,multBSR)

print(scipy.optimize.tnc.RCSTRINGS[opt_exit_mode])

plt.figure()
plt.plot(sim_nWV/lp.N_A*lp.mH2O,sim_range*1e-3,'k--')
plt.plot(nWV.profile.flatten(),nWV.range_array*1e-3,'r')
plt.plot(sol2D[1:,1]/lp.N_A*lp.mH2O,Hratio.range_array*1e-3,'g.')
plt.xlim([0,30])
plt.xlabel('Absolute Humidity [$g/m^3$]')
plt.ylabel('Altitude [km]')
plt.grid(b=True)

plt.figure()
plt.plot(sim_T,sim_range*1e-3,'k--')
plt.plot(temp.profile.flatten(),temp.range_array*1e-3,'b')
plt.plot(sol2D[1:,0],Hratio.range_array*1e-3,'g.')
#plt.xlim([150,300])
plt.xlabel('Temperature [$K$]')
plt.ylabel('Altitude [km]')
plt.grid(b=True)

plt.figure()
plt.semilogx((sim_beta_aer+sim_beta_cloud+sim_beta_mol)/sim_beta_mol,sim_range*1e-3,'k--')
plt.semilogx(bsr.profile.flatten(),bsr.range_array*1e-3,'r')
plt.semilogx(sol2D[1:,2],Hratio.range_array*1e-3,'g.')
#plt.xlim([150,300])
plt.xlabel('Particle Backscatter Coefficient [$m^{-1}sr^{-1}$]')
plt.ylabel('Altitude [km]')
plt.grid(b=True)
"""


"""
Compare assumed and actual profiles
"""

"""
HSRL Validation Process
Issue:  HSRL ratio was not agreeing with observation even when inserting the simulation
    parameters.
Resolution: molecular backscatter spectrum was not being normalized properly in the pca
    spectroscopy library.  If the frequency grid was different than the grid it trained
    on, the normalization needed to change.  There is now a normalize option in the 
    function that should be True for molecular backscatter spectrum calculation
"""

HSRLmol = td.HSRLProfile(TAct.profile.flatten(),presfit.profile.flatten(),BSRAct.profile.flatten(),rb_spec['HSRL'],Trx['HSRL Mol'],inuL['HSRL'],1.0)

HSRLcomb = td.HSRLProfile(TAct.profile.flatten(),presfit.profile.flatten(),BSRAct.profile.flatten(),rb_spec['HSRL'],Trx['HSRL Comb'],inuL['HSRL'],1.0)

HSRL_Ratio = td.HSRLProfileRatio(TAct.profile.flatten(),presfit.profile.flatten(),BSRAct.profile.flatten(),Trx['HSRL Mol'],Trx['HSRL Comb'],rb_spec['HSRL'],inuL['HSRL'],GainRatio=1.0/2.1)

TetalonHSRL = HSRL_Filter1.spectrum(sim_nu+lp.c/wavelength_list[2],InWavelength=False,aoi=Etalon_angle,transmit=True) \
            *HSRL_Filter2.spectrum(sim_nu+lp.c/wavelength_list[2],InWavelength=False,aoi=Etalon_angle,transmit=True)

beta_mol_780 = 5.45*(550.0e-9/wavelength_list[3])**4*1e-32*(sim_P/9.86923e-6)/(sim_T*lp.kB)
beta_mol = 5.45*(550.0e-9/wavelength_list[ilist])**4*1e-32*(sim_P/9.86923e-6)/(sim_T*lp.kB)
beta_aer = sim_beta_aer*wavelength_list[2]/780.24e-9
beta_cloud = sim_beta_cloud  #*wavelength_list[ilist]/780.24e-9  # No wavelength dependence in cloud
alpha_aer = beta_aer*sim_LR + beta_cloud*sim_cloud_LR
OD_aer = np.cumsum(alpha_aer)*sim_dr
OD_mol = np.cumsum(beta_mol*8*np.pi/3)*sim_dr
Taer = np.exp(-OD_aer-OD_mol)

plt.figure()
plt.semilogx(beta_mol_780,sim_range,label='actual profile')
plt.semilogx(beta_mol_sonde.profile.flatten(),beta_mol_sonde.range_array,label='assumed profile')
plt.xlabel(r'$\beta_m(R)$ [$m^{-1}sr^{-1}$]')
plt.grid(b=True)

beta_mol_sub = np.interp(TAct.range_array,sim_range,beta_mol_780)
Overlap_sub = np.interp(TAct.range_array,sim_range,Overlap)
beta_aer_sub = np.interp(TAct.range_array,sim_range,beta_aer+beta_cloud)
OD_aer_sub = np.interp(TAct.range_array,sim_range,OD_aer)
OD_mol_sub = np.interp(TAct.range_array,sim_range,OD_mol)
Taer_sub = np.exp(-OD_aer_sub-OD_mol_sub)

molBeta_nu = lp.RB_Spectrum(sim_T,sim_P,wavelength_list[2],nu=sim_nu,norm=True)
betaM_sub = spec.calc_pca_spectrum(rb_spec['HSRL'],TAct.profile.flatten(),presfit.profile.flatten())
betaM_sub_norm = np.sum(betaM_sub,axis=0)

i_plt = 10
plot_alt = TAct.range_array[i_plt]
isim = np.argmin(np.abs(sim_range-plot_alt))
plt.figure()
plt.plot(nu,betaM_sub[:,i_plt],'.')
plt.plot(sim_nu,molBeta_nu[:,isim])
plt.xlabel('Molecular Backscatter Spectrum')
plt.title('%d m'%plot_alt)

plt.figure()
plt.plot(nu,Trx['HSRL Mol'],'.')
plt.plot(sim_nu,TetalonHSRL*RbFilter)
plt.xlabel('Molecular Channel Transmission Spectrum')


T_rx_mol_sub = np.sum(Trx['HSRL Mol'][:,np.newaxis]*betaM_sub,axis=0)
T_rx_mol = np.sum(molBeta_nu*TetalonHSRL[:,np.newaxis]*RbFilter[:,np.newaxis],axis=0)

plt.figure();
plt.plot(nu,(Trx['HSRL Mol'][:,np.newaxis]*betaM_sub)[:,i_plt])
plt.plot(sim_nu,(molBeta_nu*TetalonHSRL[:,np.newaxis]*RbFilter[:,np.newaxis])[:,isim])

plt.figure()
plt.plot(T_rx_mol_sub,TAct.range_array)
plt.plot(T_rx_mol,sim_range)
plt.xlabel('Molecular Transmission Efficiency')

plt.figure()
plt.semilogx(K_list[3]*Taer_sub*Overlap_sub*beta_mol_sub*BSRAct.profile.flatten()*HSRLmol/BSRAct.range_array**2,BSRAct.range_array)
plt.semilogx(signal_list[3].profile.flatten(),signal_list[3].range_array)

plt.figure()
plt.semilogx(K_list[2]*Taer_sub*Overlap_sub*beta_mol_sub*BSRAct.profile.flatten()*HSRLcomb/BSRAct.range_array**2,BSRAct.range_array)
plt.semilogx(signal_list[2].profile.flatten(),signal_list[2].range_array)

plt.figure()
plt.plot((K_list[2]*Taer_sub*Overlap_sub*beta_mol_sub*BSRAct.profile.flatten()*HSRLcomb/BSRAct.range_array**2)/(K_list[3]*Taer_sub*Overlap_sub*beta_mol_sub*BSRAct.profile.flatten()*HSRLmol/BSRAct.range_array**2),BSRAct.range_array,label='Model from Profiles')
plt.plot(Hratio.profile.flatten(),Hratio.range_array,label='Simulation')
plt.plot(HSRL_Ratio,BSRAct.range_array,label='TD function')

"""
WV Validation Process
Issue:  It looks like the optical depth of the water vapor signal is causing
    a multiplier error between the observed signal and that obtained by the
    model.
Possible Resolution:  It looks like this multiplier is traced to the region
    between the first valid range gate and the lidar, where optical depth
    from this altitude region was not included in the calculations.  The
    question is how to integrate that into the routine.  It may need to include
    interpolation from the surface station to the first gate.
"""

molBeta_nu = lp.RB_Spectrum(sim_T,sim_P,wavelength_list[1],nu=sim_nu,norm=True)
betaM_sub = spec.calc_pca_spectrum(rb_spec['WV Online'],TAct.profile.flatten(),presfit.profile.flatten())

TetalonWVOn = Filter.spectrum(sim_nu+lp.c/wavelength_list[1],InWavelength=False,aoi=Etalon_angle,transmit=True)  

# obtain frequency resolved water vapor extinction coefficient
ext_wv = lp.WV_ExtinctionFromHITRAN(lp.c/wavelength_list[1]+sim_nu,sim_T,sim_P,nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9]),freqnorm=True).T
    
OD_wv = np.cumsum(sim_nWV[np.newaxis,:]*ext_wv,axis=1)*sim_dr  # obtain frequency resolved optical depth

T_tx = np.exp(-OD_wv[inu0,:])  # outgoing atmospheric transmission
T_rx_aer = np.exp(-OD_wv[inu0,:])*Tetalon[inu0]  # return transmission for atmosphere and etalon seen by aerosols
T_rx_mol = np.sum(molBeta_nu*TetalonWVOn[:,np.newaxis]*np.exp(-OD_wv),axis=0)  

sigS = spec.calc_pca_spectrum(abs_spec['WV Online'],TAct.profile.flatten(),presfit.profile.flatten())
r_int = np.arange(0,TAct.range_array[0],TAct.mean_dR)  # interpolated ranges between surface station and first range gate
Tint = np.interp(r_int,np.array([0,TAct.range_array[0]]),np.array([sim_T[0],TAct.profile[0,0]]))
Pint = np.interp(r_int,presfit.range_array,presfit.profile.flatten())
nWVint = np.interp(r_int,np.array([0,TAct.range_array[0]]),np.array([sim_nWV[0],nWVAct.profile[0,0]]))
ODs0 = np.sum(nWVint[np.newaxis,:]*spec.calc_pca_spectrum(abs_spec['WV Online'],Tint,Pint),axis=1)*TAct.mean_dR
#ODs0 = sim_nWV[0]*spec.calc_pca_spectrum(abs_spec['WV Online'],sim_T[0],sim_P[0])*nWVAct.range_array[0]
ODs = np.cumsum(nWVAct.profile*sigS,axis=1)*TAct.mean_dR+ODs0[:,np.newaxis]
Ts = np.exp(-ODs)
#np.sum(Trx[:,np.newaxis]*Ts*betaM,axis=0)

i_plt = 0
plot_alt = TAct.range_array[i_plt]
isim = np.argmin(np.abs(sim_range-plot_alt))

plt.figure()
plt.plot(nu,betaM_sub[:,i_plt],'.')
plt.plot(sim_nu,molBeta_nu[:,isim])
plt.xlabel('Molecular Backscatter Spectrum')
plt.title('%d m'%plot_alt)

plt.figure()
plt.plot(nu,Trx['WV Online'],'.')
plt.plot(sim_nu,TetalonWVOn)
plt.xlabel('Molecular Channel Transmission Spectrum')

plt.figure(); 
plt.plot(nu,(nWVAct.profile*sigS)[:,i_plt]); 
plt.plot(sim_nu,(sim_nWV[np.newaxis,:]*ext_wv)[:,isim],'--')
plt.title('WV Online Spectrum at %d m'%plot_alt)

T_rx_wvon_sub = np.sum(Trx['WV Online'][:,np.newaxis]*betaM_sub,axis=0)
T_rx_wvon = np.sum(molBeta_nu*TetalonWVOn[:,np.newaxis],axis=0)

plt.figure()
plt.plot(T_rx_wvon_sub,TAct.range_array)
plt.plot(T_rx_wvon,sim_range,'--')
plt.xlabel('Molecular Transmission Efficiency')