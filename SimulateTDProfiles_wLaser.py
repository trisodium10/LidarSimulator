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

sim_start = datetime.datetime.now()
ncdatafilename = sim_start.strftime('simulated_thermodynamic_DIAL_%Y%m%d_T%H%M_data.nc') # file where we write the "captured" data
ncsimfilename = sim_start.strftime('simulated_thermodynamic_DIAL_%Y%m%d_T%H%M_sim.nc')  # file where we write the simulation parameters
save_list = []  # list of simulation variables to save to the netcdf file
save_data_list = []  # list of data to save to the MCS data file

print('netcdf file names:')
print('   '+ncdatafilename)
print('   '+ncsimfilename)

use_RD = True  # simulate Rayleigh-Doppler Effect
use_shot_noise = True # simulate shot noise

Nprofiles = 2*30  # number of two second profiles

#wavelength_list = [828.3026e-9,828.203e-9,780.246119e-9,780.246119e-9,769.2339e-9,769.319768e-9]  # offline,online,Comb,Mol,online,offline  # O2 On center: 769.2339e-9
wavelength_list = [828.3026e-9,828.203e-9,780.246119e-9,780.246119e-9,769.7963e-9,770.1081e-9]  # offline,online,Comb,Mol,online,offline  # O2 On center: 769.7963
K_list = Nprofiles*np.array([0.5e16,0.5e16,1.27e16*0.3,1.15e16*0.7,0.5e16,0.5e16])  # list of multiplier constants
bg_list = Nprofiles*np.array([1.0,1.0,0.64,0.29,1.0,1.0])*5.7  # list of background levels (6 night, 150 peak day clear, 700-1000 day cloudy)
name_list = ['WV Offline','WV Online','HSRL Combined','HSRL Molecular','O2 Online','O2 Offline']  # name corresponding to each wavelength

laser_BW_list = [1000e6]*len(wavelength_list)  # laser bandwidth

Etalons = {'WV':{'center wavelength':wavelength_list[1],
                 'detune':-250e6,  # center frequency detuning in Hz
                 'efficiency':0.95,
                 'FSR':43.5e9, # free spectral range
                 'FWHM':1.0932e9, # Full width half max in Hz
                 'angle':0.0,  # angle of incidence in radians
                 'min transmission':1e-2,},  # minimum transmission (best blocking)

            'O2':{'center wavelength':wavelength_list[4],
                 'detune':-250e6,  # center frequency detuning in Hz
                 'efficiency':0.95,
                 'FSR':43.5e9, # free spectral range
                 'FWHM':1.0932e9,  # Full width half max in Hz
                 'angle':0.0,  # angle of incidence in radians
                 'min transmission':1e-2},  # minimum transmission (best blocking)

            'HSRL':[{'center wavelength':wavelength_list[2],
                 'detune':0e6,  # center frequency detuning in Hz
                 'efficiency':0.95,
                 'FSR':45e9, # free spectral range
                 'FWHM':5.7e9,  # Full width half max in Hz
                 'angle':0.0,  # angle of incidence in radians
                 'min transmission':1e-2},  # minimum transmission (best blocking)

                {'center wavelength':wavelength_list[2],
                 'detune':0e6,  # center frequency detuning in Hz
                 'efficiency':0.95,
                 'FSR':250e9, # free spectral range
                 'FWHM':15e9,  # Full width half max in Hz
                 'angle':0.0,  # angle of incidence in radians
                 'min transmission':1e-2}],  # minimum transmission (best blocking)
            }
                 

wavelen_d = dict(zip(name_list,wavelength_list))
K_d = dict(zip(name_list,K_list))
bg_d = dict(zip(name_list,bg_list))
laser_BW_d = dict(zip(name_list,laser_BW_list))

save_list.extend([{'var':Nprofiles,'varname':'Nprofiles','units':'2 minute profiles','description':''}])
for ai in range(len(wavelength_list)):
    save_list.extend([{'var':wavelength_list[ai],'varname':'wavelength_'+name_list[ai].replace(' ','_'),'units':'m','description':'wavelength of the '+name_list[ai]+' channel'}])
    save_list.extend([{'var':K_list[ai],'varname':'K_'+name_list[ai].replace(' ','_'),'units':'','description':'constant multiplier for the '+name_list[ai]+' channel'}])
    save_list.extend([{'var':bg_list[ai],'varname':'BG_'+name_list[ai].replace(' ','_'),'units':'Photon Counts','description':'Background in the '+name_list[ai]+' channel'}])
    save_list.extend([{'var':laser_BW_list[ai],'varname':'laser_bandwidth_'+name_list[ai].replace(' ','_'),'units':'','description':'Laser bandwidth in the '+name_list[ai]+' channel'}])


signal_list = [];



sim_dr = 37.5/8  # simulated range resolution
sim_LR = 35.0 # simulated lidar ratio

save_list.extend([{'var':sim_dr,'varname':'sim_dr','units':'m','description':'simulated range resolution'}])

sim_dnu = 10e6  # spacing in optical frequency space of simulation
sim_nu_max = 5e9  # edge of simulated frequency space

save_list.extend([{'var':sim_dnu,'varname':'sim_dnu','units':'Hz','description':'simulated frequency resolution'}])

sim_T0 = 300  # temp at surface in K
sim_dT1 = 6.5e-3  # lapse rate in K/m
sim_dT2 = 9.3e-3  # lapse rate in K/m
zT = 2.6e3        # transition between lapse rates

PL_laser = 150  # resolution due to laser pulse length in m

save_list.extend([{'var':PL_laser,'varname':'LaserPulseLength','units':'m','description':'simulated laser pulse length'}])

bin_res = 37.5  # range bin resolution

save_list.extend([{'var':bin_res,'varname':'bin_res','units':'m','description':'simulated MCS bin resolution in range'}])

#TauP = 8e3  # pressure decay constant in m altitude

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

save_list.extend([{'var':sim_range,'varname':'sim_range','units':'m','description':'simulated range array'}])
save_list.extend([{'var':sim_nu,'varname':'sim_nu','units':'Hz','description':'simulated relative frequency array'}])
save_list.extend([{'var':inu0,'varname':'inu0','units':'','description':'index into zero frequency in sim_nu'}])

# define laser spectrum
laser_spec = {}
for ch in laser_BW_d.keys():
    if laser_BW_d[ch] > 0:
        laser_spec[ch] = np.exp(-sim_nu**2/laser_BW_d[ch]**2)
        laser_spec_norm = np.sum(laser_spec[ch])
        if laser_spec_norm > 0:
            laser_spec[ch] = laser_spec[ch]/laser_spec_norm
        else:
            # if the spectrum is too narrow, just set it to have zero width (delta function)
            laser_spec[ch][inu0] = 1.0
            laser_BW_d[ch] = 0  
    else:
        laser_spec[ch] = np.zeros(sim_nu.size)
        laser_spec[ch][inu0] = 0
    save_list.extend([{'var':laser_spec[ch],'varname':str(ch).replace(' ','_')+'_laser_spectrum','units':'','description':'normalized laser spectrum for channel: '+str(ch)}])

# aerosol backscatter coefficient m^-1 sr^-1 at 780 nm
beta_aer_hsrl = 1e-8*np.ones(sim_range.size)
beta_aer_hsrl[np.nonzero(sim_range < 2e3)] = 1e-6
p_sim_aer = np.polyfit(sim_range,np.log10(beta_aer_hsrl),13)
sim_beta_aer = 10**(np.polyval(p_sim_aer,sim_range))
#sim_beta_aer = sim_beta_aer + np.exp(-(sim_range-cloud_alt)**2/(cloud_wid**2))

sim_beta_cloud = beta_c*np.exp(-(sim_range-cloud_alt)**2/(cloud_wid**2))

save_list.extend([{'var':sim_beta_aer,'varname':'sim_beta_aer','units':'m^-1 sr^-1','description':'simulated aerosol backscatter coefficient'}])
save_list.extend([{'var':sim_beta_cloud,'varname':'sim_beta_cloud','units':'m^-1 sr^-1','description':'simulated cloud backscatter coefficient'}])

## aerosol extinction coefficient at 780 nm
#alpha_aer_hsrl = beta_aer_hsrl*sim_LR

# absolute humidity initally in g/m^3
sim_nWVi = 20*(1-sim_range/6e3)
sim_nWVi[np.nonzero(sim_range > 4e3)] = 0.6
p_sim_wv = np.polyfit(sim_range,np.log(sim_nWVi),13)
sim_nWV = np.exp(np.polyval(p_sim_wv,sim_range))
sim_nWV = sim_nWV*lp.N_A/lp.mH2O  # convert to number/m^3

save_list.extend([{'var':sim_nWV,'varname':'sim_nWV','units':'number/m^3','description':'simulated water vapor number density'}])

# create a piecewise temperature profile
# use polyfit to add a few wiggles
i_zT = np.nonzero(sim_range > zT)
sim_T_base = sim_T0-sim_dT1*sim_range
sim_T_base[i_zT] =(sim_T0-sim_dT1*zT+sim_dT2*zT)-sim_dT2*sim_range[i_zT]
p_sim_T = np.polyfit(sim_range,sim_T_base,3)
sim_T = np.polyval(p_sim_T,sim_range)

save_list.extend([{'var':sim_T,'varname':'sim_T','units':'K','description':'simulated temperature profile'}])

#plt.figure();
#plt.plot(sim_T_base,sim_range)
#plt.plot(sim_T,sim_range)

# pressure profile in atm.
P0 = 1.0  # base pressure
sim_P = P0*(sim_T/sim_T[0])**5.2199
save_list.extend([{'var':sim_P,'varname':'sim_P','units':'atm','description':'simulated pressure profile'}])

sim_nO2 = fO2*(sim_P*101325/(lp.kB*sim_T)-sim_nWV)
save_list.extend([{'var':sim_nO2,'varname':'sim_nO2','units':'number/m^3','description':'simulated oxygen number density'}])

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
save_list.extend([{'var':Overlap,'varname':'Overlap','units':'','description':'simulated geometric overlap function'}])

#plt.figure(); 
#plt.plot(geo_corr[:,0],1.0/geo_corr[:,1])
#plt.plot(sim_range,Overlap)

# Simulate scattering in the receiver as an exponential decay
tx_scatter = np.exp(-sim_range/TauScat0)
tx_scatter = betaScat0*tx_scatter/np.sum(tx_scatter)
save_list.extend([{'var':tx_scatter,'varname':'tx_scatter','units':'Photon Counts','description':'simulated scattered light from the transitted pulse'}])



# etalon design
Etalon_angle = Etalons['WV']['angle']
Filter = FO.FP_Etalon(Etalons['WV']['FWHM'],Etalons['WV']['FSR'],lp.c/Etalons['WV']['center wavelength']+Etalons['WV']['detune'],efficiency=Etalons['WV']['efficiency'],InWavelength=False)

HSRL_Etalon_angle = Etalons['HSRL'][0]['angle']
HSRL_Filter1 = FO.FP_Etalon(Etalons['HSRL'][0]['FWHM'],Etalons['HSRL'][0]['FSR'],lp.c/Etalons['HSRL'][0]['center wavelength']+Etalons['HSRL'][0]['detune'],efficiency=Etalons['HSRL'][0]['efficiency'],InWavelength=False)
HSRL_Filter2 = FO.FP_Etalon(Etalons['HSRL'][1]['FWHM'],Etalons['HSRL'][1]['FSR'],lp.c/Etalons['HSRL'][1]['center wavelength']+Etalons['HSRL'][1]['detune'],efficiency=Etalons['HSRL'][1]['efficiency'],InWavelength=False)

O2_Etalon_angle = Etalons['O2']['angle']
O2_Filter = FO.FP_Etalon(Etalons['O2']['FWHM'],Etalons['O2']['FSR'],lp.c/Etalons['O2']['center wavelength']+Etalons['O2']['detune'],efficiency=Etalons['O2']['efficiency'],InWavelength=False)

sim_pulse = np.ones(np.round(PL_laser/sim_dr).astype(np.int))
sim_pulse = sim_pulse/sim_pulse.size

bin_func = np.ones(np.round(bin_res/sim_dr).astype(np.int))
sim_range_bin = np.arange(sim_range[-1]/bin_res)*bin_res


# Assume 97% Rb 87
Lcell = 7.2e-2  # Rb Cell length in m
Tcell = 50+274.15 # Rb Cell Temperature in K
K85, K87 = spec.RubidiumD2Spectra(Tcell,sim_nu+lp.c/wavelength_list[2],0.0)  # Calculate absorption coefficeints
RbFilter = np.exp(-Lcell*(np.sum(K87,axis=0)*0.97/0.27832+np.sum(K85,axis=0)*0.03/0.72172))  # calculate transmission accounting for isotopic fraction of the cell

save_list.extend([{'var':RbFilter,'varname':'RbFilter','units':'','description':'simulated rubidium filter transmission'}])

#spec.ExtinctionFromHITRAN(nu,TempProf,PresProf,mol,freqnorm=False,nuLim=np.array([]))
# Oxygen Spectroscopy
#ext_o2 = spec.ExtinctionFromHITRAN(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[4],sim_T[0:1],sim_P[0:1],(td.mO2*1e-3)/lp.N_A,nuLim=np.array([lp.c/770e-9,lp.c/768e-9]),freqnorm=True,filename=o2_spec_file).T
ext_o2 = spec.ExtinctionFromHITRAN(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[4],sim_T[0:1],sim_P[0:1],'O2',nuLim=np.array([lp.c/770e-9,lp.c/768e-9]),freqnorm=True).T
ext_o2off = spec.ExtinctionFromHITRAN(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[5],sim_T[0:1],sim_P[0:1],'O2',nuLim=np.array([lp.c/770e-9,lp.c/768e-9]),freqnorm=True).T
Tetalon = O2_Filter.spectrum(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[4],InWavelength=False,aoi=Etalon_angle,transmit=True)
plt.figure(); 
plt.plot(1e9*lp.c/(sim_nu+lp.c/wavelength_list[4]),laser_spec[name_list[4]]/laser_spec[name_list[4]].max(),label=name_list[4]+' Laser')
plt.plot(1e9*lp.c/(sim_nu+lp.c/wavelength_list[5]),laser_spec[name_list[5]]/laser_spec[name_list[5]].max(),label=name_list[4]+' Laser')
#plt.stem(1e9*np.array([wavelength_list[4],wavelength_list[5]]),np.ones(2),label='Laser')
plt.plot(1e9*lp.c/(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[4]),Tetalon,label='Etalon'); 
plt.plot(1e9*lp.c/(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[4]),ext_o2[:,0]/np.max(ext_o2[:,0]),label='Oxygen On'); 
plt.plot(1e9*lp.c/(np.arange(-50e9,50e9,10e6)+lp.c/wavelength_list[5]),ext_o2off[:,0]/np.max(ext_o2[:,0]),label='Oxygen Off'); 
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
save_list.extend([{'var':sim_beta_mol,'varname':'sim_beta_mol','units':'m^-1 sr^-1','description':'simulated molecular backscatter coefficient at %f nm'%(wavelength_list[2]*1e9)}])



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
    save_list.extend([{'var':BSR,'varname':'BSR_'+name_list[ilist].replace(' ','_'),'units':'','description':'simulated backscatter ratio seen by the '+name_list[ilist]+' channel'}])
    save_list.extend([{'var':Taer,'varname':'Taer_'+name_list[ilist].replace(' ','_'),'units':'','description':'simulated atmospheric aerosol/molecular transmission seen by the '+name_list[ilist]+' channel'}])
    
    
    
    # obtain the molecular backscatter spectrum
    molBeta_nu = lp.RB_Spectrum(sim_T,sim_P,wavelength_list[ilist],nu=sim_nu,norm=True)
    #plt.figure(); 
    #plt.imshow(molBeta_nu);
    
    if 'WV' in name_list[ilist]:
        # define etalon transmission
        Tetalon = Filter.spectrum(sim_nu+lp.c/wavelength_list[ilist],InWavelength=False,aoi=Etalons['WV']['angle'],transmit=True)+Etalons['WV']['min transmission']        
        
        # obtain frequency resolved water vapor extinction coefficient
#        ext_wv = lp.WV_ExtinctionFromHITRAN(lp.c/wavelength_list[ilist]+sim_nu,sim_T,sim_P,nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9]),freqnorm=True).T
        ext_wv = spec.ExtinctionFromHITRAN(lp.c/wavelength_list[ilist]+sim_nu,sim_T,sim_P,'H2O',freqnorm=True,nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9])).T
            
        
        OD_wv = np.cumsum(sim_nWV[np.newaxis,:]*ext_wv,axis=1)*sim_dr  # obtain frequency resolved optical depth
        
        if laser_BW_d[name_list[ilist]] == 0 or not use_RD:
            # Calculation for a narrow band laser
            T_tx = np.exp(-OD_wv[inu0,:])  # outgoing atmospheric transmission
            T_rx_aer = np.exp(-OD_wv[inu0,:])*Tetalon[inu0]  # return transmission for atmosphere and etalon seen by aerosols
            if use_RD:
                T_rx_mol = np.sum(molBeta_nu*Tetalon[:,np.newaxis]*np.exp(-OD_wv),axis=0)          
            else:
                T_rx_mol = np.exp(-OD_wv[inu0,:])*Tetalon[inu0] 
        else:
            # calculation for a finite bandwidth laser
            for irange in range(molBeta_nu.shape[1]):
                # convolve the molecular and laser spectrums to get the true return spectrum
                molBeta_nu[:,irange] = np.convolve(molBeta_nu[:,irange],laser_spec[name_list[ilist]],'same')
                molBeta_nu[:,irange] = molBeta_nu[:,irange]/np.sum(molBeta_nu[:,irange])  # make sure the result is still normalized
                
            T_tx = np.sum(np.exp(-OD_wv)*laser_spec[name_list[ilist]][:,np.newaxis],axis=0)  # outgoing atmospheric transmission
            T_rx_aer = np.sum(np.exp(-OD_wv)*(Tetalon*laser_spec[name_list[ilist]])[:,np.newaxis],axis=0)  # return transmission for atmosphere and etalon seen by aerosols
            T_rx_mol = np.sum(molBeta_nu*Tetalon[:,np.newaxis]*np.exp(-OD_wv),axis=0)          
                
    elif 'O2' in name_list[ilist]:
        # define etalon transmission
        Tetalon = O2_Filter.spectrum(sim_nu+lp.c/wavelength_list[ilist],InWavelength=False,aoi=Etalons['O2']['angle'],transmit=True)+Etalons['O2']['min transmission']         
        
#        ext_o2 = lp.WV_ExtinctionFromHITRAN(lp.c/wavelength_list[ilist]+sim_nu,sim_T,sim_P,nuLim=np.array([lp.c/770e-9,lp.c/768e-9]),freqnorm=True,filename=o2_spec_file).T
#        ext_o2 = spec.ExtinctionFromHITRAN(lp.c/wavelength_list[ilist]+sim_nu,sim_T,sim_P,(td.mO2*1e-3)/lp.N_A,nuLim=np.array([lp.c/770e-9,lp.c/768e-9]),freqnorm=True,filename=o2_spec_file).T
        ext_o2 = spec.ExtinctionFromHITRAN(lp.c/wavelength_list[ilist]+sim_nu,sim_T,sim_P,'O2',nuLim=np.array([lp.c/770e-9,lp.c/768e-9]),freqnorm=True).T
        
        OD_o2 = np.cumsum(sim_nO2[np.newaxis,:]*ext_o2,axis=1)*sim_dr
        if laser_BW_d[name_list[ilist]] == 0 or not use_RD:
            # Calculation for a narrow band laser
            T_tx = np.exp(-OD_o2[inu0,:])
            T_rx_aer = np.exp(-OD_o2[inu0,:])*Tetalon[inu0]
            if use_RD:
                T_rx_mol = np.sum(molBeta_nu*Tetalon[:,np.newaxis]*np.exp(-OD_o2),axis=0)
            else:
                T_rx_mol = np.exp(-OD_o2[inu0,:])*Tetalon[inu0]
            
        else:
            # calculation for a finite bandwidth laser
            for irange in range(molBeta_nu.shape[1]):
                # convolve the molecular and laser spectrums to get the true return spectrum
                molBeta_nu[:,irange] = np.convolve(molBeta_nu[:,irange],laser_spec[name_list[ilist]],'same')
                molBeta_nu[:,irange] = molBeta_nu[:,irange]/np.sum(molBeta_nu[:,irange])  # make sure the result is still normalized
                
            T_tx = np.sum(np.exp(-OD_wv)*laser_spec[name_list[ilist]][:,np.newaxis],axis=0)  # outgoing atmospheric transmission
            T_rx_aer = np.sum(np.exp(-OD_wv)*(Tetalon*laser_spec[name_list[ilist]])[:,np.newaxis],axis=0)  # return transmission for atmosphere and etalon seen by aerosols
            T_rx_mol = np.sum(molBeta_nu*Tetalon[:,np.newaxis]*np.exp(-OD_wv),axis=0) 
        
    else:
        Tetalon = (HSRL_Filter1.spectrum(sim_nu+lp.c/wavelength_list[2],InWavelength=False,aoi=Etalons['HSRL'][0]['angle'],transmit=True)+Etalons['HSRL'][0]['min transmission'])     \
            *(HSRL_Filter2.spectrum(sim_nu+lp.c/wavelength_list[2],InWavelength=False,aoi=Etalons['HSRL'][1]['angle'],transmit=True)+Etalons['HSRL'][1]['min transmission'])
        BSR_780 = BSR  # store for comparison to processing retrievals
        if laser_BW_d[name_list[ilist]] == 0:
            # calculation for narrow band laser
            if 'Molecular' in name_list[ilist]:
                T_rx_mol = np.sum(molBeta_nu*Tetalon[:,np.newaxis]*RbFilter[:,np.newaxis],axis=0)
                T_rx_aer = Tetalon[inu0]*RbFilter[inu0]
                T_tx = 1.0
            else:
                T_rx_mol = np.sum(molBeta_nu*Tetalon[:,np.newaxis],axis=0)
                T_rx_aer = Tetalon[inu0]
                T_tx = 1.0
        else:
        # calculation for a finite bandwidth laser
            for irange in range(molBeta_nu.shape[1]):
                # convolve the molecular and laser spectrums to get the true return spectrum
                molBeta_nu[:,irange] = np.convolve(molBeta_nu[:,irange],laser_spec[name_list[ilist]],'same')
                molBeta_nu[:,irange] = molBeta_nu[:,irange]/np.sum(molBeta_nu[:,irange])  # make sure the result is still normalized
            
            if 'Molecular' in name_list[ilist]:
                T_rx_mol = np.sum(molBeta_nu*Tetalon[:,np.newaxis]*RbFilter[:,np.newaxis],axis=0)
                T_rx_aer = np.sum(Tetalon*RbFilter*laser_spec[name_list[ilist]])
                T_tx = 1.0
            else:
                T_rx_mol = np.sum(molBeta_nu*Tetalon[:,np.newaxis],axis=0)
                T_rx_aer = np.sum(Tetalon*laser_spec[name_list[ilist]])
                T_tx = 1.0
        
    save_list.extend([{'var':Tetalon,'varname':'Tetalon_'+name_list[ilist].replace(' ','_'),'units':'','description':'simulated transmission through the etalon in the '+name_list[ilist]+' channel'}])
    save_list.extend([{'var':T_tx,'varname':'T_tx_'+name_list[ilist].replace(' ','_'),'units':'','description':'simulated outgoing transmission through the atmosphere in the '+name_list[ilist]+' channel'}])
    save_list.extend([{'var':T_rx_aer,'varname':'T_rx_aer_'+name_list[ilist].replace(' ','_'),'units':'','description':'simulated return transmission from aerosol scattering through the atmosphere and receiver in the '+name_list[ilist]+' channel'}])
    save_list.extend([{'var':T_rx_mol,'varname':'T_rx_mol_'+name_list[ilist].replace(' ','_'),'units':'','description':'simulated return transmission from molecular scattering through the atmosphere and receiver in the '+name_list[ilist]+' channel'}])
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

# write out the simulation parameters
for ai in range(len(save_list)):
    lp.write_var2nc(save_list[ai]['var'],save_list[ai]['varname'],ncsimfilename,units=save_list[ai]['units'],description = save_list[ai]['description'])


save_data_list.extend([{'var':sim_P[0],'varname':'P_WS','units':'atm','description':'Pressure at lidar weather station'}])
save_data_list.extend([{'var':sim_T[0],'varname':'T_WS','units':'K','description':'Temperature at lidar weather station'}])
# write out the observation parameters
for ai in range(len(save_data_list)):
    lp.write_var2nc(save_data_list[ai]['var'],save_data_list[ai]['varname'],ncdatafilename,units=save_data_list[ai]['units'],description = save_data_list[ai]['description'])

for ai, signal in enumerate(signal_list):
    signal.write2nc(ncdatafilename,write_axes=True)


lp.plotprofiles(signal_list)








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



