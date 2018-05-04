# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:27:01 2017

@author: mhayman
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:48:32 2017

@author: mhayman

Calculates the PCA needed for rapid absoprtion spectra calculations
Use sim_i to index the desired spectral region (WV/O2,online/offline)
"""

import numpy as np
import scipy.io
import LidarProfileFunctions as lp
import WVProfileFunctions as wv
import FourierOpticsLib as FO
import SpectrumLib as rb

import TDRetrievalLib as td

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import datetime

import timeit

save_data_path = '/Users/mhayman/Documents/DIAL/PCA_Rayleigh/'
save_figure_path = '/Users/mhayman/Documents/DIAL/PCA_Rayleigh/output_plots/'

wavelength_list = np.array([828.203e-9,828.3026e-9,769.2339e-9,769.319768e-9,780.24e-9,769.7963e-9,770.1081e-9])  # offline,online,Comb,Mol,online,offline
name_list = ['WV Online','WV Offline','O2 Online','O2 Offline','HSRL','O2 Online','O2 Offline']  # name corresponding to each wavelength
index_list = np.arange(len(wavelength_list))
#s_i = np.array([0,0,1,1])  # index into particular species definition

species_mass_list = np.array([lp.mH2O,td.mO2])
spec_file = ['/Users/mhayman/Documents/MATLAB/RBScatteringPCA/RB_Spectra_828203pm.mat',\
    '/Users/mhayman/Documents/MATLAB/RBScatteringPCA/RB_Spectra_828303pm.mat',\
    '/Users/mhayman/Documents/MATLAB/RBScatteringPCA/RB_Spectra_769234pm.mat',\
    '/Users/mhayman/Documents/MATLAB/RBScatteringPCA/RB_Spectra_769320pm.mat',\
    '/Users/mhayman/Documents/MATLAB/RBScatteringPCA/RB_Spectra_780240pm.mat',\
    '/Users/mhayman/Documents/MATLAB/RBScatteringPCA/RB_Spectra_769796pm.mat',\
    '/Users/mhayman/Documents/MATLAB/RBScatteringPCA/RB_Spectra_770108pm.mat']
#spec_range = [np.array([lp.c/828.5e-9,lp.c/828e-9]),np.array([lp.c/770e-9,lp.c/768e-9])]

sim_i = 6
save_results = True


save_file_name = 'RB_PCA_Data_'+name_list[sim_i].replace(' ','_') + '_%dpm'%(np.round(wavelength_list[sim_i]*1e12).astype(np.int))
run_string = datetime.datetime.today().strftime('Run On %A, %B %d, %Y, %H:%m')

Npca = 10  # number of principle components
#PtDensity = 50  # number of P and T points to evaluate (total points = PtDensity^2)

Npoly = 10

# current plan is to run PCA for multiple center points on the absoprtion line
# at +/- 5 GHz intervals.  This gives good evaluation speed and is probably
# wide enough for evaluating a single profile
# The processor would then load one PCA configuration matrix at the beginning
# and use that throughout the routine.
# We can use an indexing json file to pick out the right PCA for the 
# application.  E.g. optimization needs to be fast, general direct processing
# needs more versitility (broader bandwidth)

#sim_dnu = 10e6  # spacing in optical frequency space of simulation
#sim_nu_max = 5e9  # edge of simulated frequency space
#sim_dr = 1e3
#
#sim_T0 = 320  # temp at surface in K
#sim_dT = 9.3e-3  # lapse rate in K/m
#TauP = 8e3  # pressure decay constant in m altitude
#
##sim_range = np.arange(0,15e3,sim_dr)  # range array in m
#sim_nu = np.arange(-sim_nu_max,sim_nu_max,sim_dnu) # frequency array in Hz
#inu0 = np.argmin(np.abs(sim_nu)) # index to center of frequency array
#
#sim_P = np.linspace(1.4,1.0*np.exp(-15e3/TauP),PtDensity)
#sim_T = np.linspace(sim_T0,sim_T0-sim_dT*15e3,PtDensity)
#
##ext_wv = lp.WV_ExtinctionFromHITRAN(lp.c/wavelength_list[ilist]+sim_nu,sim_T,sim_P,nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9]),freqnorm=True).T
#
#Tm,Pm = np.meshgrid(sim_T,sim_P)
#
#ext_o2 = np.zeros((sim_nu.size,sim_P.size*sim_T.size))
#
#start_time = timeit.default_timer()
#ext_o2 = td.ExtinctionFromHITRAN(lp.c/wavelength_list[sim_i]+sim_nu,Tm.flatten(),Pm.flatten(),(species_mass_list[s_i[sim_i]]*1e-3)/lp.N_A,nuLim=spec_range[s_i[sim_i]],freqnorm=True,filename=spec_file[s_i[sim_i]]).T
#elapsed = timeit.default_timer() - start_time

MatFile = spec_file[sim_i]
matdata = scipy.io.loadmat(MatFile)
Spect = matdata['Spect'].copy()
sim_nu = (matdata['sim_nu'].copy()).flatten()
Tm = matdata['T'].copy()
Pm = matdata['P'].copy()

# normalize data
Spect = Spect/np.sum(Spect,axis=0)[np.newaxis,:]

# perform SVD    
spect_mean = np.mean(Spect,axis=1)
u,s,v = np.linalg.svd(Spect-np.mean(Spect,axis=1)[:,np.newaxis])

# truncate princpal components to the first Npca
uPCA = u[:,:Npca]

# calculate PCA weights for each T and P combination
w = np.matrix(uPCA.T)*np.matrix(Spect-np.mean(Spect,axis=1)[:,np.newaxis])

# Build bivariate polynomial power basis of order Npoly
pO = np.arange(Npoly)  # 1D power basis
pT,pP = np.meshgrid(pO,pO)  # 2D power basis with cross terms
pT = pT.flatten()
pP = pP.flatten()
Tmean = np.mean(Tm)  # Compute normalization terms for fit (mean and std)
Tstd = np.std(Tm)
Pmean = np.mean(Pm)
Pstd = np.std(Pm)
# build the polynomial matrix for the simulated cases
TPmat =((Tm.flatten()-Tmean)/Tstd)[:,np.newaxis]**pT[np.newaxis,:]*((Pm.flatten()-Pmean)/Pstd)[:,np.newaxis]**pP[np.newaxis,:]

# fit the polynomial basis to the PCA weights
wpoly = np.matrix(np.linalg.pinv(TPmat))*w.T

#save these terms:
M = uPCA*wpoly.T  # matrix for calculating the spectrum
Mavg = np.matrix(spect_mean[:,np.newaxis])  # DC term to be added back in
#sim_nu
#wavelength_list[1]
#pT
#pP
#Tmean
#Tstd
#Pmean
#Pstd
T_lims = np.array([np.min(Tm),np.max(Tm)])
P_lims = np.array([np.min(Pm),np.max(Pm)])
#Npoly
#Npca
# run_string
# report_string
# end saved terms


#RMSerror = np.zeros(Tm.size)
#maxError = np.zeros(Tm.size)
#RMSerror_poly = np.zeros(Tm.size)
#maxError_poly = np.zeros(Tm.size)
#diff_pca_poly = np.zeros(Tm.size)
#RMSdiff_pca_poly = np.zeros(Tm.size)
#for ai in range(Tm.size):
#    maxError[ai] = np.nanmax(np.abs(ext_o2[:,ai]-(np.array(uPCA*w[:,ai]).flatten()+spect_mean))/ext_o2[:,ai])
#    RMSerror[ai] = np.sqrt(np.sum((ext_o2[:,ai]-(np.array(uPCA*w[:,ai]).flatten()+spect_mean))**2))/np.sum(ext_o2[:,ai])
#    TPmat_test = ((Tm.flatten()-Tmean)/Tstd)[ai,np.newaxis]**pT[np.newaxis,:]*((Pm.flatten()-Pmean)/Pstd)[ai,np.newaxis]**pP[np.newaxis,:]
#    RMSerror_poly[ai] = np.sqrt(np.sum((ext_o2[:,ai]-np.array(M*TPmat_test.T+Mavg).flatten())**2))/np.sum(ext_o2[:,ai])
#    maxError_poly[ai] = np.nanmax(np.abs(ext_o2[:,ai]-np.array(M*TPmat_test.T+Mavg).flatten())/ext_o2[:,ai])
#    diff_pca_poly[ai] = np.max(np.abs((np.array(uPCA*w[:,ai]).flatten()+spect_mean)-np.array(M*TPmat_test.T+Mavg).flatten())/ext_o2[:,ai])
#    RMSdiff_pca_poly[ai] = np.sqrt(np.sum(((np.array(uPCA*w[:,ai]).flatten()+spect_mean)-np.array(M*TPmat_test.T+Mavg).flatten())**2))/np.sum(ext_o2[:,ai])

start_time = timeit.default_timer()
TPmat_test = ((Tm.flatten()-Tmean)/Tstd)[:,np.newaxis]**pT[np.newaxis,:]*((Pm.flatten()-Pmean)/Pstd)[:,np.newaxis]**pP[np.newaxis,:]
ext_poly = np.array(M*TPmat_test.T+Mavg)
elapsed2 = timeit.default_timer() - start_time
ext_pca = np.array(uPCA*w)+spect_mean[:,np.newaxis]

maxError = np.nanmax(np.abs(Spect-ext_pca)/Spect,axis=0)
RMSerror = np.sqrt(np.sum((Spect-ext_pca)**2,axis=0))/np.sum(Spect,axis=0)
maxError_poly = np.nanmax(np.abs(Spect-ext_poly)/Spect,axis=0)
RMSerror_poly = np.sqrt(np.sum((Spect-ext_poly)**2,axis=0))/np.sum(Spect,axis=0)
diff_pca_poly = np.max(np.abs(ext_pca-ext_poly)/Spect,axis=0)
diff_pca_poly_RMS  = np.sqrt(np.sum((ext_pca-ext_poly)**2,axis=0))/np.sum(Spect,axis=0)
diff_pca_poly_RMS_total  = np.sqrt(np.sum((ext_pca-ext_poly)**2))/np.sum(Spect)
RMSError_poly_total = np.sqrt(np.sum((Spect-ext_poly)**2))/np.sum(Spect)
RMSerror_total = np.sqrt(np.sum((Spect-ext_pca)**2))/np.sum(Spect)


elapsed = 0.0
PtDensity = np.sqrt(Pm.size)
report_string = '%s:\n'%name_list[sim_i] + 'N PCA: %d\n'%Npca \
    + 'N Poly (1D): %d\n'%Npoly \
    + '%d x %d Temperature and Pressure Grid\n'%(PtDensity,PtDensity) \
    + '+/- %.2f GHz Span\n'%(np.max(sim_nu)*1e-9) \
    + '%.2f MHz Resolution\n'%(np.mean(np.diff(sim_nu))*1e-6) \
    + 'RMS Error: %f %%\n'%(RMSError_poly_total*100) \
    + 'PCA RMS Error: %f %%\n'%(RMSerror_total*100) \
    + 'PCA-Poly RMS Difference: %f %%\n'%(diff_pca_poly_RMS_total*100) \
    + 't HITRAN:%f s\n'%elapsed \
    + 't Poly-PCA:%f s'%elapsed2

print(report_string)

#print('%s:'%name_list[sim_i])
#print('N PCA: %d'%Npca)
#print('N Poly (1D): %d'%Npoly)
#print('%d x %d Temperature and Pressure Grid'%(PtDensity,PtDensity))
#print('+/- %.2f GHz Span'%(sim_nu_max*1e-9))
#print('%.2f MHz Resolution'%(sim_dnu*1e-6))
#print('RMS Error: %f %%'%(RMSError_poly_total*100))
#print('PCA RMS Error: %f %%'%(RMSerror_total*100))
#print('PCA-Poly RMS Difference: %f %%'%(diff_pca_poly_RMS_total*100))
#print('t HITRAN:%f s'%elapsed)
#print('t Poly-PCA:%f s'%elapsed2)


w_array = np.array(w)
if save_results:
    np.savez(save_data_path+save_file_name,M=M,Mavg=Mavg,sim_nu=sim_nu, \
        wavelength=wavelength_list[sim_i],pT=pT,pP=pP,Tmean=Tmean,Tstd=Tstd,\
        Pmean=Pmean,Pstd=Pstd,T_lims=T_lims,P_lims=P_lims,Npoly=Npoly,Npca=Npca,\
        run_string=run_string,report_string=report_string)

i_weight = 9
for i_weight in range(Npca):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Tm.flatten(), Pm.flatten(), w_array[i_weight,:],c=w_array[i_weight,:]) # , c=c, marker=m
    ax.set_xlabel('T [K]')
    ax.set_ylabel('P [atm]')
    ax.set_zlabel('$w_{%d}$'%i_weight)
    if save_results:
        plt.savefig(save_figure_path+save_file_name+'_PC_weight_%d.png'%i_weight,dpi=300)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for ai in range(uPCA.shape[1]):
    ax.plot(ai*np.ones(sim_nu.size),sim_nu*1e-9,uPCA[:,ai])
ax.set_xlabel('Component Order')
ax.set_ylabel('Frequency [GHz]')
ax.set_zlabel('Amplitude')
ax.set_title('Principle Components')
if save_results:
    plt.savefig(save_figure_path+save_file_name+'_PrincipalComponents.png',dpi=300)
#plt.figure()
#plt.plot(sim_nu*1e-9,uPCA)
#plt.grid(b=True)
#plt.xlabel('Frequency [GHz]')
#plt.ylabel('Absorption Cross Section [$m^2$]')

#i_test = -1
#i_test = np.round(np.random.rand()*Tm.size).astype(np.int)
i_test_array = np.concatenate((np.array([0]),np.round(np.random.rand(3)*Tm.size).astype(np.int),np.array([-1])))
for irun in range(i_test_array.size):
    i_test = i_test_array[irun]
    TPmat_test = TPmat_test = ((Tm.flatten()-Tmean)/Tstd)[i_test,np.newaxis]**pT[np.newaxis,:]*((Pm.flatten()-Pmean)/Pstd)[i_test,np.newaxis]**pP[np.newaxis,:]
    spec_poly = np.array(M*TPmat_test.T+Mavg).flatten()
    plt.figure();
    plt.plot(sim_nu*1e-9,Spect[:,i_test])
    plt.plot(sim_nu*1e-9,np.array(uPCA*w[:,i_test]).flatten()+spect_mean,'--')
    plt.plot(sim_nu*1e-9,spec_poly,':')
    plt.grid(b=True)
    plt.legend(('HITRAN','PCA','poly-PCA'))
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Absorption Cross Section [$m^2$]')
    plt.title('%.1f K at %.2f atm.'%(Tm.flatten()[i_test],Pm.flatten()[i_test]))
    if save_results:
        plt.savefig(save_figure_path+save_file_name+'_i%d_PlotCompare.png'%i_test,dpi=300)

#plt.figure()
#plt.semilogy(RMSerror)
#plt.grid(b=True)
#plt.ylabel('Fraction RMS Error')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Tm.flatten(), Pm.flatten(), 100*RMSerror,c=RMSerror) # , c=c, marker=m
ax.set_xlabel('T')
ax.set_ylabel('P')
ax.set_zlabel('% RMS Error')
ax.set_title('PCA')
if save_results:
    plt.savefig(save_figure_path+save_file_name+'_RMS_PCA_Error.png',dpi=300)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Tm.flatten(), Pm.flatten(), 100*RMSerror_poly,c=RMSerror) # , c=c, marker=m
ax.set_xlabel('T')
ax.set_ylabel('P')
ax.set_zlabel('% RMS Error')
ax.set_title('Poly-PCA')
if save_results:
    plt.savefig(save_figure_path+save_file_name+'_RMS_Poly_PCA_Error.png',dpi=300)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Tm.flatten(), Pm.flatten(), 100*maxError,c=RMSerror) # , c=c, marker=m
ax.set_xlabel('T')
ax.set_ylabel('P')
ax.set_zlabel('Max % Error')
ax.set_title('PCA')
if save_results:
    plt.savefig(save_figure_path+save_file_name+'_Max_PCA_Error.png',dpi=300)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Tm.flatten(), Pm.flatten(), 100*maxError_poly,c=RMSerror) # , c=c, marker=m
ax.set_xlabel('T')
ax.set_ylabel('P')
ax.set_zlabel('Max % Error')
ax.set_title('Poly-PCA')
if save_results:
    plt.savefig(save_figure_path+save_file_name+'_Max_Poly_PCA_Error.png',dpi=300)