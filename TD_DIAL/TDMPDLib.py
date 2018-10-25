#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:24:23 2018

@author: mhayman
"""

import numpy as np
import LidarProfileFunctions as lp
import SpectrumLib as spec


def Build_TD_sparsa_Profiles(x,Const,dt=1.0,dR=37.5,return_params=False,params={},n_conv=0,scale={'xB':1,'xN':1,'xT':1,'xPhi':1,'xPsi':1}):
    """
    Temperature is expected in K
    Pressure is expected in atm

    dt sets the adjustment factor to convert counts to count rate needed in
    deadtime correction
    
    if return_params=True, returns the profiles of the optical parameters
    
    Constants
    Each channel:
        'WVOnline', 'WVOffline', 'MolOnline', 'MolOffline', 'CombOnline', 'CombOffline'
        each with
         'mult' - multiplier profile (C_k in documentation)
         'Trx' - receiver transmission spectrum
         'bg' - background counts
    'absPCA' - PCA definitions for absorption features with
        'WVon', 'WVoff', 'O2on', 'O2off'
    'molPCA' - PCA definitions for RB backscatter spectrum
        'O2' - the only channel this is used for
    
    'base_P' - 1D array of the weather station pressure in atm
    'base_T' - 1D array of the weather station pressure in K
    'i0' - index into center of frequency grid
        
    """
    
    try:
        BSR = params['Backscatter_Ratio']
        nWV = params['nWV']
        T = params['Temperature']
        phi = params['phi']
        psi = params['psi']
        
        tau_on = params['tau_on'] # water vapor absorption cross section
        tau_off = params['tau_off'] # water vapor absorption cross section
        sigma_on = params['sigma_on'] # oxygen absorption cross section
        sigma_off = params['sigma_off'] # oxygen absorption cross section
        
        beta = params['beta']  # molecular spectrum
        
        nO2 = params['nO2']  # oxygen number density

    except KeyError:    
        BSR = np.exp(x['xB']*scale['xB'])+1 # backscatter ratio
        nWV = np.exp(x['xN']*scale['xN'])   # water vapor number density
        T = np.cumsum(x['xT'])*scale['xT']  # temperature
        
        phi = np.exp(scale['xPhi']*x['xPhi']) # common terms at 770 nm
        psi = np.exp(scale['xPsi']*x['xPsi']) # common terms at 828 nm
        
#        tau = np.zeros(nWV.shape)
#        sigma = np.zeros(T.shape)
#        beta = np.zeros(BSR.shape)
#        for ai in range(Const['base_T'].size):
        tau_on = spec.calc_pca_T_spectrum(Const['absPCA']['WVon'],T,Const['base_T'],Const['base_P'])  # water vapor absorption cross section
        tau_on = tau_on.reshape(nWV.shape)
        tau_off = spec.calc_pca_T_spectrum(Const['absPCA']['WVoff'],T,Const['base_T'],Const['base_P'])  # water vapor absorption cross section
        tau_off = tau_off.reshape(nWV.shape)
        sigma_on = spec.calc_pca_T_spectrum(Const['absPCA']['O2on'],T,Const['base_T'],Const['base_P']) # oxygen absorption cross section
        sigma_on = sigma_on.reshape((T.shape[0],T.shape[1],Const['absPCA']['O2on']['nu_pca'].size))
        sigma_off = spec.calc_pca_T_spectrum(Const['absPCA']['O2off'],T,Const['base_T'],Const['base_P']) # oxygen absorption cross section
        sigma_off = sigma_on.reshape(T.shape)
        beta = spec.calc_pca_T_spectrum(Const['molPCA']['O2'],T,Const['base_T'],Const['base_P']) # molecular spectrum
        beta = beta.reshape((BSR.shape[0],BSR.shape[1],Const['molPCA']['O2']['nu_pca'].size))
        
        nO2 = Const['base_P'][:,np.newaxis]*T**4.2199/(lp.kB*Const['base_T'][:,np.newaxis]**5.2199)-nWV 
    
    i0 = Const['i0']  # index into transmission frequency
    
    To2on0 = np.exp(-dR*spec.fo2*np.cumsum(sigma_on[:,:,i0]*nO2,axis=1))  # center line transmission for O2
    To2on = np.exp(-dR*spec.fo2*np.cumsum(sigma_on*nO2[:,:,np.newaxis],axis=1))  # center line transmission for O2
    To2off = np.exp(-2*dR*spec.fo2*np.cumsum(sigma_off*nO2,axis=1))  # center line transmission for O2 (note factor of 2 in exponent)
    
    if 'kconv' in Const.keys():
        kconv = Const['kconv']
    elif n_conv > 0:
        kconv = np.ones((1,n_conv),dtype=np.float)/n_conv  # create convolution kernel for laser pulse
    else:
        kconv= np.ones((1,1))    
    
    
    forward_profs = {}
    for var in Const.keys():
        # get index into the gain and deadtime arrays
        if 'WVOnline' in var:
            iConst = 0
            forward_profs[var] = np.exp(x['xG'][iConst])*Const[var]['mult']*psi*np.exp(-2*dR*np.cumsum(tau_on*nWV,axis=1))+Const[var]['bg']
        elif 'WVOffline' in var:
            iConst = 1
            forward_profs[var] = np.exp(x['xG'][iConst])*Const[var]['mult']*psi*np.exp(-2*dR*np.cumsum(tau_off*nWV,axis=1))+Const[var]['bg']
        elif 'MolOnline' in var:
            iConst = 2
            forward_profs[var] = np.exp(x['xG'][iConst])*Const[var]['mult']*phi*To2on0*(Const[var]['Trx'][i0]*(BSR-1)*To2on0+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta*To2on,axis=2))+Const[var]['bg']
        elif 'MolOffline' in var:
            iConst = 3
            forward_profs[var] = np.exp(x['xG'][iConst])*Const[var]['mult']*phi*To2off*(Const[var]['Trx'][i0]*(BSR-1)+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta,axis=2))+Const[var]['bg'] 
        elif 'CombOnline' in var:
            iConst = 4
            forward_profs[var] = np.exp(x['xG'][iConst])*Const[var]['mult']*phi*To2on0*(Const[var]['Trx'][i0]*(BSR-1)*To2on0+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta*To2on,axis=2))+Const[var]['bg']
        elif 'CombOffline' in var:
            iConst = 5
            forward_profs[var] = np.exp(x['xG'][iConst])*Const[var]['mult']*phi*To2off*(Const[var]['Trx'][i0]*(BSR-1)+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta,axis=2))+Const[var]['bg']
        
            
        if 'line' in var:
            if kconv.size > 1:
                forward_profs[var] = lp.conv2d(forward_profs[var],kconv,keep_mask=False)  
    
    if return_params:
        forward_profs['Backscatter_Ratio'] = BSR.copy()
        forward_profs['nWV'] = nWV.copy()
        forward_profs['Temperature'] = T.copy()
        forward_profs['phi'] = phi.copy()
        forward_profs['psi'] = psi.copy()
        
        forward_profs['tau_on'] = tau_on.copy() # water vapor absorption cross section
        forward_profs['tau_off'] = tau_off.copy() # water vapor absorption cross section
        forward_profs['sigma_on'] = sigma_on.copy() # oxygen absorption cross section
        forward_profs['sigma_off'] = sigma_off.copy() # oxygen absorption cross section
        
        params['beta'] = beta.copy()  # molecular spectrum
        
        params['nO2'] = nO2.copy() # oxygen number density

        if kconv.size > 1:
            forward_profs['kconv'] = kconv.copy()
        
    
    return forward_profs