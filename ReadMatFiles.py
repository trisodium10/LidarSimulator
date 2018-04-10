# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:36:17 2017

@author: mhayman
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import LidarProfileFunctions as lp
import WVProfileFunctions as wv

#MatFile = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/diff_geo_corr.mat'
#DiffGeoLen = 280  # standard length of a diff_geo_corr

#MatFile = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/diff_geo_cor_170301C.mat'
#DiffGeoLen = 280*5  # standard length of a diff_geo_corr
#
#matdata = scipy.io.loadmat(MatFile)
#diff_geo_data = matdata['diff_geo_corr'].flatten()
#
## Scott blanks the bottom 3 bins of the data using Nans,
## but I don't assign a z offset to the data.
## remove the naned bins at the bottom of the profile
##iblank = np.nonzero(np.logical_not(np.isnan(diff_geo_data)))[0]  
##diff_geo_data = diff_geo_data[iblank]
#
#diff_geo_data = diff_geo_data[2:]
#
## augment the profile for missing bins to obtain a full DiffGeoLen Overlap correction
#if diff_geo_data.size < DiffGeoLen:
##    diff_geo_data = np.concatenate((np.ones(DiffGeoLen-diff_geo_data.size),diff_geo_data))
#    diff_geo_data = np.concatenate((diff_geo_data,np.ones(DiffGeoLen-diff_geo_data.size)))
#    
#
#
#np.savez('diff_geo_DLB_20170301',diff_geo_prof=diff_geo_data,Day=1,Month=3,Year=2017,HourLim=np.array([0,0]))



#### Read Rayleigh Brillouin PCA Matrix Data
#MatFile = '/h/eol/mhayman/MatlabCode/RBScatteringPCA/RB_PCA_Params.mat'
#matdata = scipy.io.loadmat(MatFile)
#M = matdata['M']
#Mavg = matdata['Mavg']
#x = matdata['x1d'].T
#
##np.savez('RB_PCA_Params',M=M,Mavg=Mavg,x=x)
#
#
#### Read DLB-Simulator Output
#
#MatFile = '/h/eol/mhayman/MatlabCode/RbHSRL/Rb_Proposed/Plots/DLB_SimulationData_01192017_Run0.mat'
#ProfSim = scipy.io.loadmat(MatFile)



MatFile = '/h/eol/mhayman/Matlab_WV_Spectrum.mat'

matdata = scipy.io.loadmat(MatFile)
sigML = matdata['sig'].flatten()
nuML = matdata['nu'].flatten()
T0ML = matdata['T0'].flatten()
P0ML = matdata['P0'].flatten()

#Tpy = Tsonde.copy()
#Tpy[0] = T0ML
#Ppy = Psonde.copy()
#Ppy[0] = P0ML

#Tpy = T0ML
#Ppy = P0ML

# T and P from Scott's example
Tpy = np.array([292.2261])
Ppy = np.array([0.9767])

nu0 = nuML.copy()
#nu0 = np.linspace(lp.c/827.0e-9,lp.c/829.5e-9,1000)
dnu = np.abs(np.mean(np.diff(nu0)))

sigWV0 = lp.WV_ExtinctionFromHITRAN(nu0,Tpy,Ppy,nuLim=np.array([lp.c/835e-9,lp.c/825e-9]),freqnorm=True) #,nuLim=np.array([lp.c/835e-9,lp.c/825e-9]))

plt.figure(); 
plt.semilogy(lp.c/nu0*1e9,sigWV0[0]);
plt.semilogy(lp.c/nuML*1e9,sigML);
plt.semilogy(np.array([828.2959,828.1960]),np.array([5.5948347e-25,2.2084761e-23])*1e-4,'ko')

#plt.figure()
#plt.plot(sigWV0[0]-sigML)