#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:24:23 2018

@author: mhayman
"""

import numpy as np
import LidarProfileFunctions as lp
import SpectrumLib as spec
import MLELidarProfileFunctions as mle

Cg = 5.2199

kB = lp.kB*9.86923e-6  # convert Boltzman constant from pressure in Pa to atm

cond_fun_default_mpd = {'xB':lambda x,y: mle.cond_pass(x,operation=y), \
                       'xT':lambda x,y: mle.cond_pass(x,operation=y), \
                       'xN':lambda x,y: mle.cond_pass(x,operation=y), \
                       'xPhi':lambda x,y: mle.cond_pass(x,operation=y), \
                       'xPsi':lambda x,y: mle.cond_pass(x,operation=y), \
                       'xG':lambda x,y: mle.cond_pass(x,operation=y), \
                       'xDT':lambda x,y: mle.cond_pass(x,operation=y)
                       }

#def Build_TD_sparsa_Profiles(x,Const,return_params=False,params={},n_conv=0,scale={'xB':1,'xN':1,'xT':1,'xPhi':1,'xPsi':1}):
#    """
#    Exponent state variables
#    Temperature is expected in K
#    Pressure is expected in atm
#
#    dt sets the adjustment factor to convert counts to count rate needed in
#    deadtime correction
#    
#    if return_params=True, returns the profiles of the optical parameters
#    
#    Constants
#    Each channel:
#        'WVOnline', 'WVOffline', 'MolOnline', 'MolOffline', 'CombOnline', 'CombOffline'
#        each with
#         'mult' - multiplier profile (C_k in documentation)
#         'Trx' - receiver transmission spectrum
#         'bg' - background counts
#         'rate_adj' - adjustment factor to convert photon counts to arrival rate
#    'absPCA' - PCA definitions for absorption features with
#        'WVon', 'WVoff', 'O2on', 'O2off'
#    'molPCA' - PCA definitions for RB backscatter spectrum
#        'O2' - the only channel this is used for
#    
#    'base_P' - 1D array of the weather station pressure in atm
#    'base_T' - 1D array of the weather station pressure in K
#    'i0' - index into center of frequency grid
#        
#    """
#    
#    try:
#        BSR = params['BSR']
#        nWV = params['nWV']
#        T = params['T']
#        phi = params['phi']
#        psi = params['psi']
#        
#        tau_on = params['tau_on'] # water vapor absorption cross section
#        tau_off = params['tau_off'] # water vapor absorption cross section
#        sigma_on = params['sigma_on'] # oxygen absorption cross section
#        sigma_off = params['sigma_off'] # oxygen absorption cross section
#        
#        beta = params['beta']  # molecular spectrum
#        
#        nO2 = params['nO2']  # atmospheric number density (not actually just nO2)
#
#    except KeyError:    
#        BSR = np.exp(x['xB']*scale['xB'])+1 # backscatter ratio
#        nWV = np.exp(x['xN']*scale['xN'])   # water vapor number density
#        T = np.cumsum(x['xT'],axis=1)*scale['xT']+Const['base_T'][:,np.newaxis]  # temperature
#        
#        phi = np.exp(scale['xPhi']*x['xPhi']) # common terms at 770 nm
#        psi = np.exp(scale['xPsi']*x['xPsi']) # common terms at 828 nm
#        
##        tau = np.zeros(nWV.shape)
##        sigma = np.zeros(T.shape)
##        beta = np.zeros(BSR.shape)
##        for ai in range(Const['base_T'].size):
#        tau_on = spec.calc_pca_T_spectrum(Const['absPCA']['WVon'],T,Const['base_T'],Const['base_P'])  # water vapor absorption cross section
#        tau_on = tau_on.reshape(nWV.shape)
#        tau_off = spec.calc_pca_T_spectrum(Const['absPCA']['WVoff'],T,Const['base_T'],Const['base_P'])  # water vapor absorption cross section
#        tau_off = tau_off.reshape(nWV.shape)
#        sigma_on = spec.calc_pca_T_spectrum(Const['absPCA']['O2on'],T,Const['base_T'],Const['base_P']) # oxygen absorption cross section
#        sigma_on = sigma_on.reshape((Const['absPCA']['O2on']['nu_pca'].size,T.shape[0],T.shape[1]))
#        sigma_on = sigma_on.transpose((1,2,0))
#        sigma_off = spec.calc_pca_T_spectrum(Const['absPCA']['O2off'],T,Const['base_T'],Const['base_P']) # oxygen absorption cross section
#        sigma_off = sigma_off.reshape(T.shape)
#        beta = spec.calc_pca_T_spectrum(Const['molPCA']['O2'],T,Const['base_T'],Const['base_P']) # molecular spectrum
#        beta = beta.reshape((Const['molPCA']['O2']['nu_pca'].size,BSR.shape[0],BSR.shape[1]))
#        beta = beta.transpose((1,2,0))
#        
#        nO2 = Const['base_P'][:,np.newaxis]*T**(Cg-1)/(kB*Const['base_T'][:,np.newaxis]**Cg)-nWV 
#    
#    dR = Const['dR']
#    i0 = Const['i0']  # index into transmission frequency
#    
#    To2on0 = np.exp(-dR*spec.fo2*np.cumsum(sigma_on[:,:,i0]*nO2,axis=1))  # center line transmission for O2
#    To2on = np.exp(-dR*spec.fo2*np.cumsum(sigma_on*nO2[:,:,np.newaxis],axis=1))  # center line transmission for O2
#    To2off = np.exp(-2*dR*spec.fo2*np.cumsum(sigma_off*nO2,axis=1))  # center line transmission for O2 (note factor of 2 in exponent)
#    
#    if 'kconv' in Const.keys():
#        kconv = Const['kconv']
#    elif n_conv > 0:
#        kconv = np.ones((1,n_conv),dtype=np.float)/n_conv  # create convolution kernel for laser pulse
#    else:
#        kconv= np.ones((1,1))    
#    
#    
#    forward_profs = {}
#    for var in Const.keys():
#        # get index into the gain and deadtime arrays
#        if 'WVOnline' in var:
#            iConst = 0
#            forward_profs[var] = np.exp(x['xG'][iConst])*Const[var]['mult']*psi*np.exp(-2*dR*np.cumsum(tau_on*nWV,axis=1))+Const[var]['bg']
#        elif 'WVOffline' in var:
#            iConst = 1
#            forward_profs[var] = np.exp(x['xG'][iConst])*Const[var]['mult']*psi*np.exp(-2*dR*np.cumsum(tau_off*nWV,axis=1))+Const[var]['bg']
#        elif 'MolOnline' in var:
#            iConst = 2
#            forward_profs[var] = np.exp(x['xG'][iConst])*Const[var]['mult']*phi*To2on0*(Const[var]['Trx'][i0]*(BSR-1)*To2on0+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta*To2on,axis=2))+Const[var]['bg']
#        elif 'MolOffline' in var:
#            iConst = 3
#            forward_profs[var] = np.exp(x['xG'][iConst])*Const[var]['mult']*phi*To2off*(Const[var]['Trx'][i0]*(BSR-1)+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta,axis=2))+Const[var]['bg'] 
#        elif 'CombOnline' in var:
#            iConst = 4
#            forward_profs[var] = np.exp(x['xG'][iConst])*Const[var]['mult']*phi*To2on0*(Const[var]['Trx'][i0]*(BSR-1)*To2on0+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta*To2on,axis=2))+Const[var]['bg']
#        elif 'CombOffline' in var:
#            iConst = 5
#            forward_profs[var] = np.exp(x['xG'][iConst])*Const[var]['mult']*phi*To2off*(Const[var]['Trx'][i0]*(BSR-1)+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta,axis=2))+Const[var]['bg']
#        
#            
#        if 'line' in var:
#            if kconv.size > 1:
#                forward_profs[var] = lp.conv2d(forward_profs[var],kconv,keep_mask=False)  
#    
#    if return_params:
#        forward_profs['BSR'] = BSR.copy()
#        forward_profs['nWV'] = nWV.copy()
#        forward_profs['T'] = T.copy()
#        forward_profs['phi'] = phi.copy()
#        forward_profs['psi'] = psi.copy()
#        
#        forward_profs['tau_on'] = tau_on.copy() # water vapor absorption cross section
#        forward_profs['tau_off'] = tau_off.copy() # water vapor absorption cross section
#        forward_profs['sigma_on'] = sigma_on.copy() # oxygen absorption cross section
#        forward_profs['sigma_off'] = sigma_off.copy() # oxygen absorption cross section
#        
#        forward_profs['beta'] = beta.copy()  # molecular spectrum
#        
#        forward_profs['nO2'] = nO2.copy() # atmospheric number density (not actually just nO2)
#
#        if kconv.size > 1:
#            forward_profs['kconv'] = kconv.copy()
#        
#    
#    return forward_profs

## Fixed condition functions
#def TD_sparsa_Error(x,fit_profs,Const,lam,weights=np.array([1]),scale={'xB':1,'xN':1,'xT':1,'xPhi':1,'xPsi':1}):  
#    
#    """
#    PTV Error of GV-HSRL profiles
#    scale={'xB':1,'xS':1,'xP':1} is deprecated
#    """
#
#    forward_profs = Build_TD_sparsa_Profiles(x,Const,return_params=True,scale=scale)
#    
#    ErrRet = 0    
#    
#    for var in lam.keys():
#        deriv = lam[var]*np.nansum(np.abs(np.diff(x[var],axis=1)))+lam[var]*np.nansum(np.abs(np.diff(x[var],axis=0)))
#        ErrRet = ErrRet + deriv
#
#    for var in fit_profs:
##        print(var)
#        if 'WVOnline' in var:
#            DT_index = 0
#        elif 'WVOffline' in var:
#            DT_index = 1
#        elif 'MolOnline' in var:
#            DT_index = 2
#        elif 'MolOffline' in var:
#            DT_index = 3
#        elif 'CombOnline' in var:
#            DT_index = 4
#        elif 'CombOffline' in var:
#            DT_index = 5
#        
##        if var == 'MolOnline':  # validation testing - separate channels
#        if not np.isnan(x['xDT'][DT_index]):
##            corrected_fit_profs = mle.deadtime_correct(fit_profs[var].profile,np.exp(x['xDT'][DT_index]),dt=Const[var]['rate_adj'])
#            corrected_fit_profs = mle.deadtime_correct(fit_profs[var].profile,x['xDT'][DT_index],dt=Const[var]['rate_adj'])
#            ErrRet += np.nansum(weights*(forward_profs[var]-corrected_fit_profs*np.log(forward_profs[var])))
#        else:
#            ErrRet += np.nansum(weights*(forward_profs[var]-fit_profs[var].profile*np.log(forward_profs[var])))
#        
#    return ErrRet



#def TD_sparsa_Error_Gradient(x,fit_profs,Const,lam,weights=np.array([1]),n_conv=0,scale={'xB':1,'xN':1,'xT':1,'xPhi':1,'xPsi':1}):
#    """
#    Analytical gradient of TD_sparsa_Error()
#    Treats all state variables as exponents
#    """
#    
#    dR = Const['dR']
#    forward_profs = Build_TD_sparsa_Profiles(x,Const,return_params=True,scale=scale)
#    
#    if 'kconv' in Const.keys():
#        kconv = Const['kconv']
#    elif n_conv > 0:
#        kconv = np.ones((1,n_conv),dtype=np.float)/n_conv  # create convolution kernel for laser pulse
#    else:
#        kconv= np.ones((1,1)) 
#    
#
#    # precalculate some quantites that are used in more than one
#    # forward model
#    i0 = Const['i0']
#    beta,dbetadT = spec.calc_pca_T_spectrum_w_deriv(Const['molPCA']['O2'],forward_profs['T'],Const['base_T'],Const['base_P']) # molecular spectrum
#    beta = beta.reshape((Const['molPCA']['O2']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
#    beta = beta.transpose((1,2,0))
#    dbetadT = dbetadT.reshape((Const['molPCA']['O2']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
#    dbetadT = dbetadT.transpose((1,2,0))
#    
#    abs_spec = {}
#    for var in Const['absPCA']:
#        abs_spec[var] = {}
#        sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA'][var],forward_profs['T'],Const['base_T'],Const['base_P'])
#        if var == 'O2on':
#            sig = sig.reshape((Const['absPCA']['O2on']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
#            abs_spec[var]['sig'] = sig.transpose((1,2,0))
#            dsigdT = dsigdT.reshape((Const['absPCA']['O2on']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
#            abs_spec[var]['dsigdT'] = dsigdT.transpose((1,2,0))
#            abs_spec[var]['Tatm0'] = np.exp(-dR*spec.fo2*np.cumsum(abs_spec[var]['sig'][:,:,i0]*forward_profs['nO2'],axis=1))
#            abs_spec[var]['Tatm'] = np.exp(-dR*spec.fo2*np.cumsum(abs_spec[var]['sig']*forward_profs['nO2'][:,:,np.newaxis],axis=1))
#        else:
#            abs_spec[var]['sig'] = sig.reshape(forward_profs['T'].shape)
#            abs_spec[var]['dsigdT'] = dsigdT.reshape(forward_profs['T'].shape)
#            
#        if var == 'O2off':
#            abs_spec[var]['Tatm0'] = np.exp(-dR*spec.fo2*np.cumsum(abs_spec[var]['sig']*forward_profs['nO2'],axis=1))
#        elif 'WV' in var:
#            abs_spec[var]['Tatm0'] = np.exp(-dR*np.cumsum(abs_spec[var]['sig']*forward_profs['nWV'],axis=1))
#            
#    #obtain models without nonlinear response or background
#    sig_profs = {}
#    e0 = {}
##    e_dt = {}
#    grad0 = {}
#    
#    # gradient components of each atmospheric variable
#    gradErr = {}
#    for var in x.keys():
#        gradErr[var] = np.zeros(x[var].shape)
#    
#    for var in fit_profs.keys():
#        sig_profs[var] = forward_profs[var]-Const[var]['bg']
#        
#        # Channel Order:
#        # 'WVOnline', 'WVOffline', 'MolOnline', 'MolOffline', 'CombOnline', 'CombOffline'
#        
#        if 'WVOnline' in var:
#            ichan = 0
#        elif 'WVOffline' in var:
#            ichan = 1
#        elif 'MolOnline' in var:
#            ichan = 2
#        elif 'MolOffline' in var:
#            ichan = 3
#        elif 'CombOnline' in var:
#            ichan = 4
#        elif 'CombOffline' in var:
#            ichan = 5
#        
#        deadtime = np.exp(x['xDT'][ichan])
#        Gain = np.exp(x['xG'][ichan])
#            
#        if not np.isnan(deadtime):
#            corrected_fit_profs = mle.deadtime_correct(fit_profs[var].profile,deadtime,dt=Const[var]['rate_adj'])
#        else:
#            corrected_fit_profs = fit_profs[var].profile.copy()
#            
#        # useful definitions for gradient calculations
#        e0[var] = (1-corrected_fit_profs/forward_profs[var])  # error function derivative
##        e_dt[var] = dt[:,np.newaxis]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2  # dead time derivative
##        grad0[var] = dR*(np.sum(e0[var]*sig_profs[var],axis=1)[:,np.newaxis]-np.cumsum(e0[var]*sig_profs[var],axis=1))
#        
#        grad0[var] = np.cumsum((e0[var]*sig_profs[var])[:,::-1],axis=1)[:,::-1]
#    
#        # dead time gradient term
#        if 'xDT' in gradErr.keys():
#            gradErr['xDT'][ichan] = -np.nansum(Const[var]['rate_adj']*fit_profs[var].profile**2/(Const[var]['rate_adj']-fit_profs[var].profile*deadtime)**2*deadtime*np.log(forward_profs[var]))
#        
#        gradErr['xG'][ichan] = np.nansum(e0[var]*sig_profs[var]) 
#            
#        #  and var == 'MolOnline' # validation testing - separate channels
#        if 'xT' in gradErr.keys():
#            if not 'WV' in var and 'Online' in var:
#                sig = abs_spec['O2on']['sig']
#                dsigdT = abs_spec['O2on']['dsigdT']
#                Tatm = abs_spec['O2on']['Tatm']
#                Tatm0 = abs_spec['O2on']['Tatm0']
##                sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA']['O2on'],forward_profs['T'],Const['base_T'],Const['base_P']) # oxygen absorption cross section
##                sig = sig.reshape((Const['absPCA']['O2on']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
##                sig = sig.transpose((1,2,0))
##                dsigdT = dsigdT.reshape((Const['absPCA']['O2on']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
##                dsigdT = dsigdT.transpose((1,2,0))
#
#                # temperature gradient for optical depth
##                grad_o = scale['xT']*np.cumsum(np.cumsum((dR*spec.fo2*dsigdT*(forward_profs['nO2']-forward_profs['nWV'])[:,:,np.newaxis] \
##                            +spec.fo2*sig*(Cg-1)*Const['base_P'][:,np.newaxis,np.newaxis]/(kB*Const['base_T'][:,np.newaxis,np.newaxis]**Cg)*forward_profs['T'][:,:,np.newaxis])[:,::-1,:],axis=1),axis=1)[:,::-1,:]
#                grad_o = scale['xT']*dR*spec.fo2*(dsigdT*forward_profs['nO2'][:,:,np.newaxis] \
#                            +sig*(Cg-1)*Const['base_P'][:,np.newaxis,np.newaxis]/(kB*Const['base_T'][:,np.newaxis,np.newaxis]**Cg)*(forward_profs['T']**(Cg-2))[:,:,np.newaxis])
#                
##                Tatm0 = np.exp(-dR*spec.fo2*np.cumsum(sig[:,:,i0]*forward_profs['nO2'],axis=1))
##                Tatm = np.exp(-dR*spec.fo2*np.cumsum(sig*forward_profs['nO2'],axis=1))
#                
#                # compute each summed term separately
#                c = -e0[var]*sig_profs[var]
#                a = grad_o[:,:,i0]
#                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                c = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0**2*Const[var]['Trx'][i0]*(forward_profs['BSR']-1)
#                a = grad_o[:,:,i0]
#                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                c1 = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0
#                c2 = Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta*Tatm
#                a = grad_o
#                gradErr['xT']+= np.cumsum(np.nansum(a[:,::-1,:]*np.cumsum(c2[:,::-1,:]*c1[:,::-1,np.newaxis],axis=1),axis=2),axis=1)[:,::-1]
#                
#                c1 = e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0
#                c2 = Const[var]['Trx'][np.newaxis,np.newaxis,:]*dbetadT*Tatm
#                gradErr['xT']+=scale['xT']*np.cumsum((c1*np.nansum(c2,axis=2))[:,::-1],axis=1)[:,::-1]
#                
#                """
#                c = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0 \
#                    *2*Const[var]['Trx'][i0]*Tatm0*(forward_profs['BSR']-1)
#                a = grad_o[:,:,i0]
#                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                c = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0 \
#                    *np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta,axis=2)
##                a = grad_o[:,:,i0]
#                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                c1 = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0
#                c2 = Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta
#                a = grad_o
#                gradErr['xT']+= np.cumsum(np.nansum(a[:,::-1,:]*np.cumsum(c2[:,::-1,:],axis=1),axis=2)*np.cumsum(c1[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                gradErr['xT']+=scale['xT']*np.cumsum((e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0 \
#                    *np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*dbetadT,axis=2))[:,::-1],axis=1)[:,::-1]
#                """    
#                    
#                
##                gradErr['xT']+= -e0[var]*Gain*Const[var]['mult']*forward_profs['Phi']*Tatm0 \
##                        *(2*grad_o[:,:,i0]*Const[var]['Trx'][i0]*Tatm0*(forward_profs['BSR']-1) \
##                        +grad_o[:,:,i0]*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta,axis=2) \
##                        +np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*(beta*grad_o-dbetadT),axis=2))
#            
#            elif 'MolOffline' in var:
#                sig = abs_spec['O2off']['sig']
#                dsigdT = abs_spec['O2off']['dsigdT']
#                Tatm0 = abs_spec['O2off']['Tatm0']
#                
##                sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA']['O2off'],forward_profs['T'],Const['base_T'],Const['base_P']) # oxygen absorption cross section
##                sig = sig.reshape(forward_profs['T'].shape)
##                dsigdT = dsigdT.reshape(forward_profs['T'].shape)
#                grad_o = scale['xT']*dR*spec.fo2*(dsigdT*forward_profs['nO2'] \
#                            +sig*(Cg-1)*Const['base_P'][:,np.newaxis]/(kB*Const['base_T'][:,np.newaxis]**Cg)*forward_profs['T']**(Cg-2))
#                
##                Tatm0 = np.exp(-dR*spec.fo2*np.cumsum(sig*forward_profs['nO2'],axis=1))
#                
#                gradErr['xT']+= np.cumsum(grad_o[:,::-1]*np.cumsum((-2*e0[var]*sig_profs[var])[:,::-1],axis=1),axis=1)[:,::-1]
#                gradErr['xT']+= scale['xT']*np.cumsum((e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0**2*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*dbetadT,axis=2))[:,::-1],axis=1)[:,::-1]
#                
##                gradErr['xT']+=e0[var]*(sig_profs[var]*(-2*grad_o) \
##                        +Gain*Const[var]['mult']*forward_profs['Phi']*Tatm0**2*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*dbetadT,axis=2))
#                        
#            elif 'CombOffline' in var:
#                sig = abs_spec['O2off']['sig']
#                dsigdT = abs_spec['O2off']['dsigdT']
#                Tatm0 = abs_spec['O2off']['Tatm0']
#                
##                sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA']['O2off'],forward_profs['T'],Const['base_T'],Const['base_P']) # oxygen absorption cross section
##                sig = sig.reshape(forward_profs['T'].shape)
##                dsigdT = dsigdT.reshape(forward_profs['T'].shape)
#                grad_o = scale['xT']*dR*spec.fo2*(dsigdT*forward_profs['nO2'] \
#                            +sig*(Cg-1)*Const['base_P'][:,np.newaxis]/(kB*Const['base_T'][:,np.newaxis]**Cg)*forward_profs['T']**(Cg-2))
#                
##                Tatm0 = np.exp(-dR*spec.fo2*np.cumsum(sig*forward_profs['nO2'],axis=1))
##                gradErr['xT']+= np.cumsum(grad_o[:,::-1]*np.cumsum((-2*e0[var]*sig_profs[var])[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                c = -2*e0[var]*sig_profs[var]
#                a = grad_o
#                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                gradErr['xT']+= scale['xT']*np.cumsum((e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0**2*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*dbetadT,axis=2))[:,::-1],axis=1)[:,::-1]
##                c = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0**2
#                
#                
#            elif 'WV' in var:
#                if 'Online' in var:
#                    spec_def = 'WVon'
#                elif 'Offline' in var:
#                    spec_def = 'WVoff'
#                
#                sig = abs_spec[spec_def]['sig']
#                dsigdT = abs_spec[spec_def]['dsigdT']
#                Tatm0 = abs_spec[spec_def]['Tatm0']
#                    
##                sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA'][spec_def],forward_profs['T'],Const['base_T'],Const['base_P']) # oxygen absorption cross section
##                sig = sig.reshape(forward_profs['T'].shape)
##                dsigdT = dsigdT.reshape(forward_profs['T'].shape)
#                
#                c = -2*dR*e0[var]*sig_profs[var]
#                a = scale['xT']*dsigdT*forward_profs['nWV']
#                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                
#                
#        if 'xN' in gradErr.keys():
#            if not 'WV' in var and 'Online' in var:
#                
#                sig = abs_spec['O2on']['sig']
#                dsigdT = abs_spec['O2on']['dsigdT']
#                Tatm = abs_spec['O2on']['Tatm']
#                Tatm0 = abs_spec['O2on']['Tatm0']
#                
#                # water vapor gradient for optical depth    
#                grad_o = dR*spec.fo2*sig*forward_profs['nWV'][:,:,np.newaxis]*scale['xN']
#                
#                
#                
#                gradErr['xN']+= -2*grad_o[:,:,i0]*np.cumsum((e0[var]*Gain*Const[var]['mult'] \
#                       *forward_profs['phi']*Tatm0**2*Const[var]['Trx'][i0]*(forward_profs['BSR']-1))[:,::-1],axis=1)[:,::-1]
#                
#                gradErr['xN']+= grad_o[:,:,i0]*np.cumsum((-e0[var]*Gain*Const[var]['mult'] \
#                       *forward_profs['phi']*Tatm0*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta,axis=2))[:,::-1],axis=1)[:,::-1]
#                
#                gradErr['xN']+= np.nansum(grad_o*np.cumsum((Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta)[:,::-1,:],axis=1)[:,::-1,:],axis=2) \
#                    *np.cumsum((-e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0)[:,::-1],axis=1)[:,::-1]
#                
##                # original
##                gradErr['xN']+= -e0[var]*Gain*Const[var]['mult']*forward_profs['Phi']*Tatm0 \
##                        *(2*grad_o[:,:,i0]*Const[var]['Trx'][i0]*Tatm0*(forward_profs['BSR']-1) \
##                        +grad_o[:,:,i0]*np.nansum(Const[var]['Trx']*Tatm*beta,axis=2)+np.nansum(Const[var]['Trx']*Tatm*beta*grad_o,axis=2))
#            elif 'CombOffline' in var:
#                
#                sig = abs_spec['O2off']['sig']
#                dsigdT = abs_spec['O2off']['dsigdT']
#                Tatm0 = abs_spec['O2off']['Tatm0']
#                
#                grad_o = -dR*spec.fo2*sig*forward_profs['nWV']*scale['xN']
#                
#                gradErr['xN']+= 2*grad_o*grad0[var]
#                
#            elif 'WV' in var:
#                if 'Online' in var:
#                    spec_def = 'WVon'
#                elif 'Offline' in var:
#                    spec_def = 'WVoff'
#                
#                sig = abs_spec[spec_def]['sig']
#                dsigdT = abs_spec[spec_def]['dsigdT']
#                Tatm0 = abs_spec[spec_def]['Tatm0']
#                
#                grad_o = -dR*sig*forward_profs['nWV']*scale['xN']
#                
#                gradErr['xN']+= 2*grad_o*grad0[var]
#                
#        
#        if 'xB' in gradErr.keys():
#            if not 'WV' in var: 
#                if 'Online' in var:
#                    gradErr['xB']+= e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*abs_spec['O2on']['Tatm0']**2*Const[var]['Trx'][i0]*(forward_profs['BSR']-1)*scale['xB']
#                elif 'Offline' in var:
#                    gradErr['xB']+= e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*abs_spec['O2off']['Tatm0']**2*Const[var]['Trx'][i0]*(forward_profs['BSR']-1)*scale['xB']
#                    
#        if 'xPhi' in gradErr.keys():
#            if 'Comb' in var or 'Mol' in var:
#                gradErr['xPhi']+=e0[var]*sig_profs[var]*scale['xPhi']
#        
#        if 'xPsi' in gradErr.keys():
#            if 'WV' in var:
#                gradErr['xPsi']+=e0[var]*sig_profs[var]*scale['xPsi']
#            
#            
#    if kconv.size > 1:
#        if 'xT' in gradErr.keys():
#            gradErr['xT'] = lp.conv2d(gradErr['xT'],kconv,keep_mask=False)
#        if 'xB' in gradErr.keys():
#            gradErr['xB'] = lp.conv2d(gradErr['xB'],kconv,keep_mask=False)
#        if 'xN' in gradErr.keys():
#            gradErr['xN'] = lp.conv2d(gradErr['xN'],kconv,keep_mask=False)
#        if 'xPhi' in gradErr.keys():
#            gradErr['xPhi'] = lp.conv2d(gradErr['xPhi'],kconv,keep_mask=False)
#        if 'xPsi' in gradErr.keys():
#            gradErr['xPsi'] = lp.conv2d(gradErr['xPsi'],kconv,keep_mask=False)
#    
#    return gradErr
   
    
#
#   Fixed linear functions
#def Build_TD_sparsa_Profiles(x,Const,return_params=False,params={},n_conv=0,scale={'xB':1,'xN':1,'xT':1,'xPhi':1,'xPsi':1}):
#    """
#    Linear state variables for nWV, BSR and Temp
#    Temperature is expected in K
#    Pressure is expected in atm
#
#    dt sets the adjustment factor to convert counts to count rate needed in
#    deadtime correction
#    
#    if return_params=True, returns the profiles of the optical parameters
#    
#    Constants
#    Each channel:
#        'WVOnline', 'WVOffline', 'MolOnline', 'MolOffline', 'CombOnline', 'CombOffline'
#        each with
#         'mult' - multiplier profile (C_k in documentation)
#         'Trx' - receiver transmission spectrum
#         'bg' - background counts
#         'rate_adj' - adjustment factor to convert photon counts to arrival rate
#    'absPCA' - PCA definitions for absorption features with
#        'WVon', 'WVoff', 'O2on', 'O2off'
#    'molPCA' - PCA definitions for RB backscatter spectrum
#        'O2' - the only channel this is used for
#    
#    'base_P' - 1D array of the weather station pressure in atm
#    'base_T' - 1D array of the weather station pressure in K
#    'i0' - index into center of frequency grid
#        
#    """
#    
#    try:
#        BSR = params['BSR']
#        nWV = params['nWV']
#        T = params['T']
#        phi = params['phi']
#        psi = params['psi']
#        
#        tau_on = params['tau_on'] # water vapor absorption cross section
#        tau_off = params['tau_off'] # water vapor absorption cross section
#        sigma_on = params['sigma_on'] # oxygen absorption cross section
#        sigma_off = params['sigma_off'] # oxygen absorption cross section
#        
#        beta = params['beta']  # molecular spectrum
#        
#        nO2 = params['nO2']  # atmospheric number density (not actually just nO2)
#
#    except KeyError:    
#        BSR = x['xB']*scale['xB'] # backscatter ratio
#        nWV = x['xN']*scale['xN']   # water vapor number density
#        T = np.cumsum(x['xT'],axis=1)*scale['xT']+Const['base_T'][:,np.newaxis]  # temperature
#        
#        phi = np.exp(scale['xPhi']*x['xPhi']) # common terms at 770 nm
#        psi = np.exp(scale['xPsi']*x['xPsi']) # common terms at 828 nm
#        
##        tau = np.zeros(nWV.shape)
##        sigma = np.zeros(T.shape)
##        beta = np.zeros(BSR.shape)
##        for ai in range(Const['base_T'].size):
#        tau_on = spec.calc_pca_T_spectrum(Const['absPCA']['WVon'],T,Const['base_T'],Const['base_P'])  # water vapor absorption cross section
#        tau_on = tau_on.reshape(nWV.shape)
#        tau_off = spec.calc_pca_T_spectrum(Const['absPCA']['WVoff'],T,Const['base_T'],Const['base_P'])  # water vapor absorption cross section
#        tau_off = tau_off.reshape(nWV.shape)
#        sigma_on = spec.calc_pca_T_spectrum(Const['absPCA']['O2on'],T,Const['base_T'],Const['base_P']) # oxygen absorption cross section
#        sigma_on = sigma_on.reshape((Const['absPCA']['O2on']['nu_pca'].size,T.shape[0],T.shape[1]))
#        sigma_on = sigma_on.transpose((1,2,0))
#        sigma_off = spec.calc_pca_T_spectrum(Const['absPCA']['O2off'],T,Const['base_T'],Const['base_P']) # oxygen absorption cross section
#        sigma_off = sigma_off.reshape(T.shape)
#        beta = spec.calc_pca_T_spectrum(Const['molPCA']['O2'],T,Const['base_T'],Const['base_P']) # molecular spectrum
#        beta = beta.reshape((Const['molPCA']['O2']['nu_pca'].size,BSR.shape[0],BSR.shape[1]))
#        beta = beta.transpose((1,2,0))
#        
#        nO2 = Const['base_P'][:,np.newaxis]*T**(Cg-1)/(kB*Const['base_T'][:,np.newaxis]**Cg)-nWV 
#    
#    dR = Const['dR']
#    i0 = Const['i0']  # index into transmission frequency
#    
#    To2on0 = np.exp(-dR*spec.fo2*np.cumsum(sigma_on[:,:,i0]*nO2,axis=1))  # center line transmission for O2
#    To2on = np.exp(-dR*spec.fo2*np.cumsum(sigma_on*nO2[:,:,np.newaxis],axis=1))  # center line transmission for O2
#    To2off = np.exp(-2*dR*spec.fo2*np.cumsum(sigma_off*nO2,axis=1))  # center line transmission for O2 (note factor of 2 in exponent)
#    
#    if 'kconv' in Const.keys():
#        kconv = Const['kconv']
#    elif n_conv > 0:
#        kconv = np.ones((1,n_conv),dtype=np.float)/n_conv  # create convolution kernel for laser pulse
#    else:
#        kconv= np.ones((1,1))    
#    
#    
#    forward_profs = {}
#    for var in Const.keys():
#        # get index into the gain and deadtime arrays
#        if 'WVOnline' in var:
#            iConst = 0
#            forward_profs[var] = x['xG'][iConst]*Const[var]['mult']*psi*np.exp(-2*dR*np.cumsum(tau_on*nWV,axis=1))+Const[var]['bg']
#        elif 'WVOffline' in var:
#            iConst = 1
#            forward_profs[var] = x['xG'][iConst]*Const[var]['mult']*psi*np.exp(-2*dR*np.cumsum(tau_off*nWV,axis=1))+Const[var]['bg']
#        elif 'MolOnline' in var:
#            iConst = 2
#            forward_profs[var] = x['xG'][iConst]*Const[var]['mult']*phi*To2on0*(Const[var]['Trx'][i0]*(BSR-1)*To2on0+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta*To2on,axis=2))+Const[var]['bg']
#        elif 'MolOffline' in var:
#            iConst = 3
#            forward_profs[var] = x['xG'][iConst]*Const[var]['mult']*phi*To2off*(Const[var]['Trx'][i0]*(BSR-1)+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta,axis=2))+Const[var]['bg'] 
#        elif 'CombOnline' in var:
#            iConst = 4
#            forward_profs[var] = x['xG'][iConst]*Const[var]['mult']*phi*To2on0*(Const[var]['Trx'][i0]*(BSR-1)*To2on0+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta*To2on,axis=2))+Const[var]['bg']
#        elif 'CombOffline' in var:
#            iConst = 5
#            forward_profs[var] = x['xG'][iConst]*Const[var]['mult']*phi*To2off*(Const[var]['Trx'][i0]*(BSR-1)+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta,axis=2))+Const[var]['bg']
#        
#            
#        if 'line' in var:
#            if kconv.size > 1:
#                forward_profs[var] = lp.conv2d(forward_profs[var],kconv,keep_mask=False)  
#    
#    if return_params:
#        forward_profs['BSR'] = BSR.copy()
#        forward_profs['nWV'] = nWV.copy()
#        forward_profs['T'] = T.copy()
#        forward_profs['phi'] = phi.copy()
#        forward_profs['psi'] = psi.copy()
#        
#        forward_profs['tau_on'] = tau_on.copy() # water vapor absorption cross section
#        forward_profs['tau_off'] = tau_off.copy() # water vapor absorption cross section
#        forward_profs['sigma_on'] = sigma_on.copy() # oxygen absorption cross section
#        forward_profs['sigma_off'] = sigma_off.copy() # oxygen absorption cross section
#        
#        forward_profs['beta'] = beta.copy()  # molecular spectrum
#        
#        forward_profs['nO2'] = nO2.copy() # atmospheric number density (not actually just nO2)
#
#        if kconv.size > 1:
#            forward_profs['kconv'] = kconv.copy()
#        
#    
#    return forward_profs
#
#def TD_sparsa_Error_Gradient(x,fit_profs,Const,lam,weights=np.array([1]),n_conv=0,scale={'xB':1,'xN':1,'xT':1,'xPhi':1,'xPsi':1}):
#    """
#    Analytical gradient of TD_sparsa_Error()
#    Treats nWV, BSR and T as linear variables
#    """
#    
#    dR = Const['dR']
#    forward_profs = Build_TD_sparsa_Profiles(x,Const,return_params=True,scale=scale)
#    
#    if 'kconv' in Const.keys():
#        kconv = Const['kconv']
#    elif n_conv > 0:
#        kconv = np.ones((1,n_conv),dtype=np.float)/n_conv  # create convolution kernel for laser pulse
#    else:
#        kconv= np.ones((1,1)) 
#    
#
#    # precalculate some quantites that are used in more than one
#    # forward model
#    i0 = Const['i0']
#    beta,dbetadT = spec.calc_pca_T_spectrum_w_deriv(Const['molPCA']['O2'],forward_profs['T'],Const['base_T'],Const['base_P']) # molecular spectrum
#    beta = beta.reshape((Const['molPCA']['O2']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
#    beta = beta.transpose((1,2,0))
#    dbetadT = dbetadT.reshape((Const['molPCA']['O2']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
#    dbetadT = dbetadT.transpose((1,2,0))
#    
#    abs_spec = {}
#    for var in Const['absPCA']:
#        abs_spec[var] = {}
#        sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA'][var],forward_profs['T'],Const['base_T'],Const['base_P'])
#        if var == 'O2on':
#            sig = sig.reshape((Const['absPCA']['O2on']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
#            abs_spec[var]['sig'] = sig.transpose((1,2,0))
#            dsigdT = dsigdT.reshape((Const['absPCA']['O2on']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
#            abs_spec[var]['dsigdT'] = dsigdT.transpose((1,2,0))
#            abs_spec[var]['Tatm0'] = np.exp(-dR*spec.fo2*np.cumsum(abs_spec[var]['sig'][:,:,i0]*forward_profs['nO2'],axis=1))
#            abs_spec[var]['Tatm'] = np.exp(-dR*spec.fo2*np.cumsum(abs_spec[var]['sig']*forward_profs['nO2'][:,:,np.newaxis],axis=1))
#        else:
#            abs_spec[var]['sig'] = sig.reshape(forward_profs['T'].shape)
#            abs_spec[var]['dsigdT'] = dsigdT.reshape(forward_profs['T'].shape)
#            
#        if var == 'O2off':
#            abs_spec[var]['Tatm0'] = np.exp(-dR*spec.fo2*np.cumsum(abs_spec[var]['sig']*forward_profs['nO2'],axis=1))
#        elif 'WV' in var:
#            abs_spec[var]['Tatm0'] = np.exp(-dR*np.cumsum(abs_spec[var]['sig']*forward_profs['nWV'],axis=1))
#            
#    #obtain models without nonlinear response or background
#    sig_profs = {}
#    e0 = {}
##    e_dt = {}
#    grad0 = {}
#    
#    # gradient components of each atmospheric variable
#    gradErr = {}
#    for var in x.keys():
#        gradErr[var] = np.zeros(x[var].shape)
#    
#    for var in fit_profs.keys():
#        sig_profs[var] = forward_profs[var]-Const[var]['bg']
#        
#        # Channel Order:
#        # 'WVOnline', 'WVOffline', 'MolOnline', 'MolOffline', 'CombOnline', 'CombOffline'
#        
#        if 'WVOnline' in var:
#            ichan = 0
#        elif 'WVOffline' in var:
#            ichan = 1
#        elif 'MolOnline' in var:
#            ichan = 2
#        elif 'MolOffline' in var:
#            ichan = 3
#        elif 'CombOnline' in var:
#            ichan = 4
#        elif 'CombOffline' in var:
#            ichan = 5
#        
##        deadtime = np.exp(x['xDT'][ichan])
##        Gain = np.exp(x['xG'][ichan])
#        
#        deadtime = x['xDT'][ichan]
#        Gain = x['xG'][ichan]
#            
#        if not np.isnan(deadtime):
#            corrected_fit_profs = mle.deadtime_correct(fit_profs[var].profile,deadtime,dt=Const[var]['rate_adj'])
#        else:
#            corrected_fit_profs = fit_profs[var].profile.copy()
#            
#        # useful definitions for gradient calculations
#        e0[var] = (1-corrected_fit_profs/forward_profs[var])  # error function derivative
##        e_dt[var] = dt[:,np.newaxis]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2  # dead time derivative
##        grad0[var] = dR*(np.sum(e0[var]*sig_profs[var],axis=1)[:,np.newaxis]-np.cumsum(e0[var]*sig_profs[var],axis=1))
#        
#        grad0[var] = np.cumsum((e0[var]*sig_profs[var])[:,::-1],axis=1)[:,::-1]
#    
#        # dead time gradient term
#        if 'xDT' in gradErr.keys():
#            gradErr['xDT'][ichan] = -np.nansum(Const[var]['rate_adj']*fit_profs[var].profile**2/(Const[var]['rate_adj']-fit_profs[var].profile*deadtime)**2*np.log(forward_profs[var]))
##            gradErr['xDT'][ichan] = -np.nansum(Const[var]['rate_adj']*fit_profs[var].profile**2/(Const[var]['rate_adj']-fit_profs[var].profile*deadtime)**2*deadtime*np.log(forward_profs[var]))
#        
##        gradErr['xG'][ichan] = np.nansum(e0[var]*sig_profs[var]) 
#        gradErr['xG'][ichan] = np.nansum(e0[var]*sig_profs[var]/Gain) 
#            
#        #  and var == 'MolOnline' # validation testing - separate channels
#        if 'xT' in gradErr.keys():
#            if not 'WV' in var and 'Online' in var:
#                sig = abs_spec['O2on']['sig']
#                dsigdT = abs_spec['O2on']['dsigdT']
#                Tatm = abs_spec['O2on']['Tatm']
#                Tatm0 = abs_spec['O2on']['Tatm0']
##                sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA']['O2on'],forward_profs['T'],Const['base_T'],Const['base_P']) # oxygen absorption cross section
##                sig = sig.reshape((Const['absPCA']['O2on']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
##                sig = sig.transpose((1,2,0))
##                dsigdT = dsigdT.reshape((Const['absPCA']['O2on']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
##                dsigdT = dsigdT.transpose((1,2,0))
#
#                # temperature gradient for optical depth
##                grad_o = scale['xT']*np.cumsum(np.cumsum((dR*spec.fo2*dsigdT*(forward_profs['nO2']-forward_profs['nWV'])[:,:,np.newaxis] \
##                            +spec.fo2*sig*(Cg-1)*Const['base_P'][:,np.newaxis,np.newaxis]/(kB*Const['base_T'][:,np.newaxis,np.newaxis]**Cg)*forward_profs['T'][:,:,np.newaxis])[:,::-1,:],axis=1),axis=1)[:,::-1,:]
#                grad_o = scale['xT']*dR*spec.fo2*(dsigdT*forward_profs['nO2'][:,:,np.newaxis] \
#                            +sig*(Cg-1)*Const['base_P'][:,np.newaxis,np.newaxis]/(kB*Const['base_T'][:,np.newaxis,np.newaxis]**Cg)*(forward_profs['T']**(Cg-2))[:,:,np.newaxis])
#                
##                Tatm0 = np.exp(-dR*spec.fo2*np.cumsum(sig[:,:,i0]*forward_profs['nO2'],axis=1))
##                Tatm = np.exp(-dR*spec.fo2*np.cumsum(sig*forward_profs['nO2'],axis=1))
#                
#                # compute each summed term separately
#                c = -e0[var]*sig_profs[var]
#                a = grad_o[:,:,i0]
#                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                c = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0**2*Const[var]['Trx'][i0]*(forward_profs['BSR']-1)
#                a = grad_o[:,:,i0]
#                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                c1 = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0
#                c2 = Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta*Tatm
#                a = grad_o
#                gradErr['xT']+= np.cumsum(np.nansum(a[:,::-1,:]*np.cumsum(c2[:,::-1,:]*c1[:,::-1,np.newaxis],axis=1),axis=2),axis=1)[:,::-1]
#                
#                c1 = e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0
#                c2 = Const[var]['Trx'][np.newaxis,np.newaxis,:]*dbetadT*Tatm
#                gradErr['xT']+=scale['xT']*np.cumsum((c1*np.nansum(c2,axis=2))[:,::-1],axis=1)[:,::-1]
#                
#                """
#                c = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0 \
#                    *2*Const[var]['Trx'][i0]*Tatm0*(forward_profs['BSR']-1)
#                a = grad_o[:,:,i0]
#                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                c = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0 \
#                    *np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta,axis=2)
##                a = grad_o[:,:,i0]
#                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                c1 = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0
#                c2 = Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta
#                a = grad_o
#                gradErr['xT']+= np.cumsum(np.nansum(a[:,::-1,:]*np.cumsum(c2[:,::-1,:],axis=1),axis=2)*np.cumsum(c1[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                gradErr['xT']+=scale['xT']*np.cumsum((e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0 \
#                    *np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*dbetadT,axis=2))[:,::-1],axis=1)[:,::-1]
#                """    
#                    
#                
##                gradErr['xT']+= -e0[var]*Gain*Const[var]['mult']*forward_profs['Phi']*Tatm0 \
##                        *(2*grad_o[:,:,i0]*Const[var]['Trx'][i0]*Tatm0*(forward_profs['BSR']-1) \
##                        +grad_o[:,:,i0]*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta,axis=2) \
##                        +np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*(beta*grad_o-dbetadT),axis=2))
#            
#            elif 'MolOffline' in var:
#                sig = abs_spec['O2off']['sig']
#                dsigdT = abs_spec['O2off']['dsigdT']
#                Tatm0 = abs_spec['O2off']['Tatm0']
#                
##                sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA']['O2off'],forward_profs['T'],Const['base_T'],Const['base_P']) # oxygen absorption cross section
##                sig = sig.reshape(forward_profs['T'].shape)
##                dsigdT = dsigdT.reshape(forward_profs['T'].shape)
#                grad_o = scale['xT']*dR*spec.fo2*(dsigdT*forward_profs['nO2'] \
#                            +sig*(Cg-1)*Const['base_P'][:,np.newaxis]/(kB*Const['base_T'][:,np.newaxis]**Cg)*forward_profs['T']**(Cg-2))
#                
##                Tatm0 = np.exp(-dR*spec.fo2*np.cumsum(sig*forward_profs['nO2'],axis=1))
#                
#                gradErr['xT']+= np.cumsum(grad_o[:,::-1]*np.cumsum((-2*e0[var]*sig_profs[var])[:,::-1],axis=1),axis=1)[:,::-1]
#                gradErr['xT']+= scale['xT']*np.cumsum((e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0**2*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*dbetadT,axis=2))[:,::-1],axis=1)[:,::-1]
#                
##                gradErr['xT']+=e0[var]*(sig_profs[var]*(-2*grad_o) \
##                        +Gain*Const[var]['mult']*forward_profs['Phi']*Tatm0**2*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*dbetadT,axis=2))
#                        
#            elif 'CombOffline' in var:
#                sig = abs_spec['O2off']['sig']
#                dsigdT = abs_spec['O2off']['dsigdT']
#                Tatm0 = abs_spec['O2off']['Tatm0']
#                
##                sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA']['O2off'],forward_profs['T'],Const['base_T'],Const['base_P']) # oxygen absorption cross section
##                sig = sig.reshape(forward_profs['T'].shape)
##                dsigdT = dsigdT.reshape(forward_profs['T'].shape)
#                grad_o = scale['xT']*dR*spec.fo2*(dsigdT*forward_profs['nO2'] \
#                            +sig*(Cg-1)*Const['base_P'][:,np.newaxis]/(kB*Const['base_T'][:,np.newaxis]**Cg)*forward_profs['T']**(Cg-2))
#                
##                Tatm0 = np.exp(-dR*spec.fo2*np.cumsum(sig*forward_profs['nO2'],axis=1))
##                gradErr['xT']+= np.cumsum(grad_o[:,::-1]*np.cumsum((-2*e0[var]*sig_profs[var])[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                c = -2*e0[var]*sig_profs[var]
#                a = grad_o
#                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                gradErr['xT']+= scale['xT']*np.cumsum((e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0**2*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*dbetadT,axis=2))[:,::-1],axis=1)[:,::-1]
##                c = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0**2
#                
#                
#            elif 'WV' in var:
#                if 'Online' in var:
#                    spec_def = 'WVon'
#                elif 'Offline' in var:
#                    spec_def = 'WVoff'
#                
#                sig = abs_spec[spec_def]['sig']
#                dsigdT = abs_spec[spec_def]['dsigdT']
#                Tatm0 = abs_spec[spec_def]['Tatm0']
#                    
##                sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA'][spec_def],forward_profs['T'],Const['base_T'],Const['base_P']) # oxygen absorption cross section
##                sig = sig.reshape(forward_profs['T'].shape)
##                dsigdT = dsigdT.reshape(forward_profs['T'].shape)
#                
#                c = -2*dR*e0[var]*sig_profs[var]
#                a = scale['xT']*dsigdT*forward_profs['nWV']
#                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
#                
#                
#                
#        if 'xN' in gradErr.keys():
#            if not 'WV' in var and 'Online' in var:
#                
#                sig = abs_spec['O2on']['sig']
#                dsigdT = abs_spec['O2on']['dsigdT']
#                Tatm = abs_spec['O2on']['Tatm']
#                Tatm0 = abs_spec['O2on']['Tatm0']
#                
#                # water vapor gradient for optical depth    
#                grad_o = dR*spec.fo2*sig*scale['xN']
#                
#                
#                
#                gradErr['xN']+= -2*grad_o[:,:,i0]*np.cumsum((e0[var]*Gain*Const[var]['mult'] \
#                       *forward_profs['phi']*Tatm0**2*Const[var]['Trx'][i0]*(forward_profs['BSR']-1))[:,::-1],axis=1)[:,::-1]
#                
#                gradErr['xN']+= grad_o[:,:,i0]*np.cumsum((-e0[var]*Gain*Const[var]['mult'] \
#                       *forward_profs['phi']*Tatm0*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta,axis=2))[:,::-1],axis=1)[:,::-1]
#                
#                gradErr['xN']+= np.nansum(grad_o*np.cumsum((Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta)[:,::-1,:],axis=1)[:,::-1,:],axis=2) \
#                    *np.cumsum((-e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0)[:,::-1],axis=1)[:,::-1]
#                
##                # original
##                gradErr['xN']+= -e0[var]*Gain*Const[var]['mult']*forward_profs['Phi']*Tatm0 \
##                        *(2*grad_o[:,:,i0]*Const[var]['Trx'][i0]*Tatm0*(forward_profs['BSR']-1) \
##                        +grad_o[:,:,i0]*np.nansum(Const[var]['Trx']*Tatm*beta,axis=2)+np.nansum(Const[var]['Trx']*Tatm*beta*grad_o,axis=2))
#            elif 'CombOffline' in var:
#                
#                sig = abs_spec['O2off']['sig']
#                dsigdT = abs_spec['O2off']['dsigdT']
#                Tatm0 = abs_spec['O2off']['Tatm0']
#                
#                grad_o = -dR*spec.fo2*sig*scale['xN']
#                
#                gradErr['xN']+= 2*grad_o*grad0[var]
#                
#            elif 'WV' in var:
#                if 'Online' in var:
#                    spec_def = 'WVon'
#                elif 'Offline' in var:
#                    spec_def = 'WVoff'
#                
#                sig = abs_spec[spec_def]['sig']
#                dsigdT = abs_spec[spec_def]['dsigdT']
#                Tatm0 = abs_spec[spec_def]['Tatm0']
#                
#                grad_o = -dR*sig*scale['xN']
#                
#                gradErr['xN']+= 2*grad_o*grad0[var]
#                
#        
#        if 'xB' in gradErr.keys():
#            if not 'WV' in var: 
#                if 'Online' in var:
#                    gradErr['xB']+= e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*abs_spec['O2on']['Tatm0']**2*Const[var]['Trx'][i0]*scale['xB']
#                elif 'Offline' in var:
#                    gradErr['xB']+= e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*abs_spec['O2off']['Tatm0']**2*Const[var]['Trx'][i0]*scale['xB']
#                    
#        if 'xPhi' in gradErr.keys():
#            if 'Comb' in var or 'Mol' in var:
#                gradErr['xPhi']+=e0[var]*sig_profs[var]*scale['xPhi']
#        
#        if 'xPsi' in gradErr.keys():
#            if 'WV' in var:
#                gradErr['xPsi']+=e0[var]*sig_profs[var]*scale['xPsi']
#            
#            
#    if kconv.size > 1:
#        if 'xT' in gradErr.keys():
#            gradErr['xT'] = lp.conv2d(gradErr['xT'],kconv,keep_mask=False)
#        if 'xB' in gradErr.keys():
#            gradErr['xB'] = lp.conv2d(gradErr['xB'],kconv,keep_mask=False)
#        if 'xN' in gradErr.keys():
#            gradErr['xN'] = lp.conv2d(gradErr['xN'],kconv,keep_mask=False)
#        if 'xPhi' in gradErr.keys():
#            gradErr['xPhi'] = lp.conv2d(gradErr['xPhi'],kconv,keep_mask=False)
#        if 'xPsi' in gradErr.keys():
#            gradErr['xPsi'] = lp.conv2d(gradErr['xPsi'],kconv,keep_mask=False)
#    
#    return gradErr

def TD_sparsa_Error(x,fit_profs,Const,lam,weights=np.array([1]),cond_fun=cond_fun_default_mpd):  
    
    """
    PTV Error of GV-HSRL profiles
    scale={'xB':1,'xS':1,'xP':1} is deprecated
    """

    forward_profs = Build_TD_sparsa_Profiles(x,Const,return_params=True,cond_fun=cond_fun)
    deadtime = cond_fun['xDT'](x['xDT'],'normal')
    ErrRet = 0    
    
    for var in lam.keys():
        deriv = lam[var]*np.nansum(np.abs(np.diff(x[var],axis=1)))+lam[var]*np.nansum(np.abs(np.diff(x[var],axis=0)))
        ErrRet = ErrRet + deriv

    for var in fit_profs:
#        print(var)
        if 'WVOnline' in var:
            DT_index = 0
        elif 'WVOffline' in var:
            DT_index = 1
        elif 'MolOnline' in var:
            DT_index = 2
        elif 'MolOffline' in var:
            DT_index = 3
        elif 'CombOnline' in var:
            DT_index = 4
        elif 'CombOffline' in var:
            DT_index = 5
        
#        if var == 'MolOnline':  # validation testing - separate channels
        if not np.isnan(x['xDT'][DT_index]):
#            corrected_fit_profs = mle.deadtime_correct(fit_profs[var].profile,np.exp(x['xDT'][DT_index]),dt=Const[var]['rate_adj'])
            corrected_fit_profs = mle.deadtime_correct(fit_profs[var].profile,deadtime[DT_index],dt=Const[var]['rate_adj'])
            ErrRet += np.nansum(weights*(forward_profs[var]-corrected_fit_profs*np.log(forward_profs[var])))
        else:
            ErrRet += np.nansum(weights*(forward_profs[var]-fit_profs[var].profile*np.log(forward_profs[var])))
        
    return ErrRet


def Build_TD_sparsa_Profiles(x,Const,return_params=False,params={},n_conv=0,cond_fun=cond_fun_default_mpd):
    """
    Linear state variables for nWV, BSR and Temp
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
         'rate_adj' - adjustment factor to convert photon counts to arrival rate
    'absPCA' - PCA definitions for absorption features with
        'WVon', 'WVoff', 'O2on', 'O2off'
    'molPCA' - PCA definitions for RB backscatter spectrum
        'O2' - the only channel this is used for
    
    'base_P' - 1D array of the weather station pressure in atm
    'base_T' - 1D array of the weather station pressure in K
    'i0' - index into center of frequency grid
        
    """
    
    try:
        BSR = params['BSR']
        nWV = params['nWV']
        T = params['T']
        phi = params['phi']
        psi = params['psi']
        
        tau_on = params['tau_on'] # water vapor absorption cross section
        tau_off = params['tau_off'] # water vapor absorption cross section
        sigma_on = params['sigma_on'] # oxygen absorption cross section
        sigma_off = params['sigma_off'] # oxygen absorption cross section
        
        beta = params['beta']  # molecular spectrum
        
        nO2 = params['nO2']  # atmospheric number density (not actually just nO2)
        
        Gain = params['Gain']

    except KeyError:    
        BSR = cond_fun['xB'](x['xB'],'normal') # backscatter ratio
        nWV = cond_fun['xN'](x['xN'],'normal')   # water vapor number density
        T = np.cumsum(cond_fun['xT'](x['xT'],'normal'),axis=1)+Const['base_T'][:,np.newaxis]  # temperature
        
        phi = cond_fun['xPhi'](x['xPhi'],'normal') # common terms at 770 nm
        psi = cond_fun['xPsi'](x['xPsi'],'normal') # common terms at 828 nm
        
        Gain = cond_fun['xG'](x['xG'],'normal')
        
#        tau = np.zeros(nWV.shape)
#        sigma = np.zeros(T.shape)
#        beta = np.zeros(BSR.shape)
#        for ai in range(Const['base_T'].size):
        tau_on = spec.calc_pca_T_spectrum(Const['absPCA']['WVon'],T,Const['base_T'],Const['base_P'])  # water vapor absorption cross section
        tau_on = tau_on.reshape(nWV.shape)
        tau_off = spec.calc_pca_T_spectrum(Const['absPCA']['WVoff'],T,Const['base_T'],Const['base_P'])  # water vapor absorption cross section
        tau_off = tau_off.reshape(nWV.shape)
        sigma_on = spec.calc_pca_T_spectrum(Const['absPCA']['O2on'],T,Const['base_T'],Const['base_P']) # oxygen absorption cross section
        sigma_on = sigma_on.reshape((Const['absPCA']['O2on']['nu_pca'].size,T.shape[0],T.shape[1]))
        sigma_on = sigma_on.transpose((1,2,0))
        sigma_off = spec.calc_pca_T_spectrum(Const['absPCA']['O2off'],T,Const['base_T'],Const['base_P']) # oxygen absorption cross section
        sigma_off = sigma_off.reshape(T.shape)
        beta = spec.calc_pca_T_spectrum(Const['molPCA']['O2'],T,Const['base_T'],Const['base_P']) # molecular spectrum
        beta = beta.reshape((Const['molPCA']['O2']['nu_pca'].size,BSR.shape[0],BSR.shape[1]))
        beta = beta.transpose((1,2,0))
        
        nO2 = Const['base_P'][:,np.newaxis]*T**(Cg-1)/(kB*Const['base_T'][:,np.newaxis]**Cg)-nWV 
    
    dR = Const['dR']
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
            forward_profs[var] = Gain[iConst]*Const[var]['mult']*psi*np.exp(-2*dR*np.cumsum(tau_on*nWV,axis=1))+Const[var]['bg']
        elif 'WVOffline' in var:
            iConst = 1
            forward_profs[var] = Gain[iConst]*Const[var]['mult']*psi*np.exp(-2*dR*np.cumsum(tau_off*nWV,axis=1))+Const[var]['bg']
        elif 'MolOnline' in var:
            iConst = 2
            forward_profs[var] = Gain[iConst]*Const[var]['mult']*phi*To2on0*(Const[var]['Trx'][i0]*(BSR-1)*To2on0+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta*To2on,axis=2))+Const[var]['bg']
        elif 'MolOffline' in var:
            iConst = 3
            forward_profs[var] = Gain[iConst]*Const[var]['mult']*phi*To2off*(Const[var]['Trx'][i0]*(BSR-1)+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta,axis=2))+Const[var]['bg'] 
        elif 'CombOnline' in var:
            iConst = 4
            forward_profs[var] = Gain[iConst]*Const[var]['mult']*phi*To2on0*(Const[var]['Trx'][i0]*(BSR-1)*To2on0+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta*To2on,axis=2))+Const[var]['bg']
        elif 'CombOffline' in var:
            iConst = 5
            forward_profs[var] = Gain[iConst]*Const[var]['mult']*phi*To2off*(Const[var]['Trx'][i0]*(BSR-1)+np.sum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta,axis=2))+Const[var]['bg']
        
            
        if 'line' in var:
            if kconv.size > 1:
                forward_profs[var] = lp.conv2d(forward_profs[var],kconv,keep_mask=False)  
    
    if return_params:
        forward_profs['BSR'] = BSR.copy()
        forward_profs['nWV'] = nWV.copy()
        forward_profs['T'] = T.copy()
        forward_profs['phi'] = phi.copy()
        forward_profs['psi'] = psi.copy()
        
        forward_profs['tau_on'] = tau_on.copy() # water vapor absorption cross section
        forward_profs['tau_off'] = tau_off.copy() # water vapor absorption cross section
        forward_profs['sigma_on'] = sigma_on.copy() # oxygen absorption cross section
        forward_profs['sigma_off'] = sigma_off.copy() # oxygen absorption cross section
        
        forward_profs['beta'] = beta.copy()  # molecular spectrum
        
        forward_profs['nO2'] = nO2.copy() # atmospheric number density (not actually just nO2)

        forward_profs['Gain'] = Gain.copy()  # channel gains
        
        if kconv.size > 1:
            forward_profs['kconv'] = kconv.copy()
        
    
    return forward_profs

def TD_sparsa_Error_Gradient(x,fit_profs,Const,lam,weights=np.array([1]),n_conv=0,cond_fun=cond_fun_default_mpd):
    """
    Analytical gradient of TD_sparsa_Error()
    Treats nWV, BSR and T as linear variables
    """
    
    dR = Const['dR']
    forward_profs = Build_TD_sparsa_Profiles(x,Const,return_params=True,cond_fun=cond_fun)
    
    if 'kconv' in Const.keys():
        kconv = Const['kconv']
    elif n_conv > 0:
        kconv = np.ones((1,n_conv),dtype=np.float)/n_conv  # create convolution kernel for laser pulse
    else:
        kconv= np.ones((1,1)) 
    

    # precalculate some quantites that are used in more than one
    # forward model
    i0 = Const['i0']
    beta,dbetadT = spec.calc_pca_T_spectrum_w_deriv(Const['molPCA']['O2'],forward_profs['T'],Const['base_T'],Const['base_P']) # molecular spectrum
    beta = beta.reshape((Const['molPCA']['O2']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
    beta = beta.transpose((1,2,0))
    dbetadT = dbetadT.reshape((Const['molPCA']['O2']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
    dbetadT = dbetadT.transpose((1,2,0))
    
    abs_spec = {}
    for var in Const['absPCA']:
        abs_spec[var] = {}
        sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA'][var],forward_profs['T'],Const['base_T'],Const['base_P'])
        if var == 'O2on':
            sig = sig.reshape((Const['absPCA']['O2on']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
            abs_spec[var]['sig'] = sig.transpose((1,2,0))
            dsigdT = dsigdT.reshape((Const['absPCA']['O2on']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
            abs_spec[var]['dsigdT'] = dsigdT.transpose((1,2,0))
            abs_spec[var]['Tatm0'] = np.exp(-dR*spec.fo2*np.cumsum(abs_spec[var]['sig'][:,:,i0]*forward_profs['nO2'],axis=1))
            abs_spec[var]['Tatm'] = np.exp(-dR*spec.fo2*np.cumsum(abs_spec[var]['sig']*forward_profs['nO2'][:,:,np.newaxis],axis=1))
        else:
            abs_spec[var]['sig'] = sig.reshape(forward_profs['T'].shape)
            abs_spec[var]['dsigdT'] = dsigdT.reshape(forward_profs['T'].shape)
            
        if var == 'O2off':
            abs_spec[var]['Tatm0'] = np.exp(-dR*spec.fo2*np.cumsum(abs_spec[var]['sig']*forward_profs['nO2'],axis=1))
        elif 'WV' in var:
            abs_spec[var]['Tatm0'] = np.exp(-dR*np.cumsum(abs_spec[var]['sig']*forward_profs['nWV'],axis=1))
           
    deadtime_set = cond_fun['xDT'](x['xDT'],'normal')
    ddeadtime_set = cond_fun['xDT'](x['xDT'],'derivative')
    Gain_set = cond_fun['xG'](x['xG'],'normal')
    dGain_set = cond_fun['xG'](x['xG'],'derivative')
    #obtain models without nonlinear response or background
    sig_profs = {}
    e0 = {}
#    e_dt = {}
    grad0 = {}
    
    # gradient components of each atmospheric variable
    gradErr = {}
    for var in x.keys():
        gradErr[var] = np.zeros(x[var].shape)
    
    for var in fit_profs.keys():
        sig_profs[var] = forward_profs[var]-Const[var]['bg']
        
        # Channel Order:
        # 'WVOnline', 'WVOffline', 'MolOnline', 'MolOffline', 'CombOnline', 'CombOffline'
        
        if 'WVOnline' in var:
            ichan = 0
        elif 'WVOffline' in var:
            ichan = 1
        elif 'MolOnline' in var:
            ichan = 2
        elif 'MolOffline' in var:
            ichan = 3
        elif 'CombOnline' in var:
            ichan = 4
        elif 'CombOffline' in var:
            ichan = 5
        
#        deadtime = np.exp(x['xDT'][ichan])
#        Gain = np.exp(x['xG'][ichan])
        
        deadtime = deadtime_set[ichan]
        Gain = Gain_set[ichan]
            
        if not np.isnan(deadtime):
            corrected_fit_profs = mle.deadtime_correct(fit_profs[var].profile,deadtime,dt=Const[var]['rate_adj'])
        else:
            corrected_fit_profs = fit_profs[var].profile.copy()
            
        # useful definitions for gradient calculations
        e0[var] = (1-corrected_fit_profs/forward_profs[var])  # error function derivative
#        e_dt[var] = dt[:,np.newaxis]**2/(dt[:,np.newaxis]+lin_profs[var]*deadtime)**2  # dead time derivative
#        grad0[var] = dR*(np.sum(e0[var]*sig_profs[var],axis=1)[:,np.newaxis]-np.cumsum(e0[var]*sig_profs[var],axis=1))
        
        grad0[var] = np.cumsum((e0[var]*sig_profs[var])[:,::-1],axis=1)[:,::-1]
    
        # dead time gradient term
        if 'xDT' in gradErr.keys():
            gradErr['xDT'][ichan] = -np.nansum(Const[var]['rate_adj']*fit_profs[var].profile**2/(Const[var]['rate_adj']-fit_profs[var].profile*deadtime)**2*ddeadtime_set[ichan]*np.log(forward_profs[var]))
#            gradErr['xDT'][ichan] = -np.nansum(Const[var]['rate_adj']*fit_profs[var].profile**2/(Const[var]['rate_adj']-fit_profs[var].profile*deadtime)**2*deadtime*np.log(forward_profs[var]))
        
#        gradErr['xG'][ichan] = np.nansum(e0[var]*sig_profs[var]) 
        gradErr['xG'][ichan] = np.nansum(e0[var]*sig_profs[var]/Gain*dGain_set[ichan]) 
            
        #  and var == 'MolOnline' # validation testing - separate channels
        if 'xT' in gradErr.keys():
            if not 'WV' in var and 'Online' in var:
                sig = abs_spec['O2on']['sig']
                dsigdT = abs_spec['O2on']['dsigdT']
                Tatm = abs_spec['O2on']['Tatm']
                Tatm0 = abs_spec['O2on']['Tatm0']
#                sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA']['O2on'],forward_profs['T'],Const['base_T'],Const['base_P']) # oxygen absorption cross section
#                sig = sig.reshape((Const['absPCA']['O2on']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
#                sig = sig.transpose((1,2,0))
#                dsigdT = dsigdT.reshape((Const['absPCA']['O2on']['nu_pca'].size,forward_profs['T'].shape[0],forward_profs['T'].shape[1]))
#                dsigdT = dsigdT.transpose((1,2,0))

                # temperature gradient for optical depth
#                grad_o = scale['xT']*np.cumsum(np.cumsum((dR*spec.fo2*dsigdT*(forward_profs['nO2']-forward_profs['nWV'])[:,:,np.newaxis] \
#                            +spec.fo2*sig*(Cg-1)*Const['base_P'][:,np.newaxis,np.newaxis]/(kB*Const['base_T'][:,np.newaxis,np.newaxis]**Cg)*forward_profs['T'][:,:,np.newaxis])[:,::-1,:],axis=1),axis=1)[:,::-1,:]
                grad_o = cond_fun['xT'](x['xT'],'derivative')[:,:,np.newaxis]*dR*spec.fo2*(dsigdT*forward_profs['nO2'][:,:,np.newaxis] \
                            +sig*(Cg-1)*Const['base_P'][:,np.newaxis,np.newaxis]/(kB*Const['base_T'][:,np.newaxis,np.newaxis]**Cg)*(forward_profs['T']**(Cg-2))[:,:,np.newaxis])
                
#                Tatm0 = np.exp(-dR*spec.fo2*np.cumsum(sig[:,:,i0]*forward_profs['nO2'],axis=1))
#                Tatm = np.exp(-dR*spec.fo2*np.cumsum(sig*forward_profs['nO2'],axis=1))
                
                # compute each summed term separately
                c = -e0[var]*sig_profs[var]
                a = grad_o[:,:,i0]
                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
                
                c = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0**2*Const[var]['Trx'][i0]*(forward_profs['BSR']-1)
                a = grad_o[:,:,i0]
                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
                
                c1 = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0
                c2 = Const[var]['Trx'][np.newaxis,np.newaxis,:]*beta*Tatm
                a = grad_o
                gradErr['xT']+= np.cumsum(np.nansum(a[:,::-1,:]*np.cumsum(c2[:,::-1,:]*c1[:,::-1,np.newaxis],axis=1),axis=2),axis=1)[:,::-1]
                
                c1 = e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0
                c2 = Const[var]['Trx'][np.newaxis,np.newaxis,:]*dbetadT*Tatm
                gradErr['xT']+=cond_fun['xT'](x['xT'],'derivative')*np.cumsum((c1*np.nansum(c2,axis=2))[:,::-1],axis=1)[:,::-1]
                
                """
                c = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0 \
                    *2*Const[var]['Trx'][i0]*Tatm0*(forward_profs['BSR']-1)
                a = grad_o[:,:,i0]
                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
                
                c = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0 \
                    *np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta,axis=2)
#                a = grad_o[:,:,i0]
                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
                
                c1 = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0
                c2 = Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta
                a = grad_o
                gradErr['xT']+= np.cumsum(np.nansum(a[:,::-1,:]*np.cumsum(c2[:,::-1,:],axis=1),axis=2)*np.cumsum(c1[:,::-1],axis=1),axis=1)[:,::-1]
                
                gradErr['xT']+=scale['xT']*np.cumsum((e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0 \
                    *np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*dbetadT,axis=2))[:,::-1],axis=1)[:,::-1]
                """    
                    
                
#                gradErr['xT']+= -e0[var]*Gain*Const[var]['mult']*forward_profs['Phi']*Tatm0 \
#                        *(2*grad_o[:,:,i0]*Const[var]['Trx'][i0]*Tatm0*(forward_profs['BSR']-1) \
#                        +grad_o[:,:,i0]*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta,axis=2) \
#                        +np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*(beta*grad_o-dbetadT),axis=2))
            
            elif 'MolOffline' in var:
                sig = abs_spec['O2off']['sig']
                dsigdT = abs_spec['O2off']['dsigdT']
                Tatm0 = abs_spec['O2off']['Tatm0']
                
#                sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA']['O2off'],forward_profs['T'],Const['base_T'],Const['base_P']) # oxygen absorption cross section
#                sig = sig.reshape(forward_profs['T'].shape)
#                dsigdT = dsigdT.reshape(forward_profs['T'].shape)
                grad_o = cond_fun['xT'](x['xT'],'derivative')*dR*spec.fo2*(dsigdT*forward_profs['nO2'] \
                            +sig*(Cg-1)*Const['base_P'][:,np.newaxis]/(kB*Const['base_T'][:,np.newaxis]**Cg)*forward_profs['T']**(Cg-2))
                
#                Tatm0 = np.exp(-dR*spec.fo2*np.cumsum(sig*forward_profs['nO2'],axis=1))
                
                gradErr['xT']+= np.cumsum(grad_o[:,::-1]*np.cumsum((-2*e0[var]*sig_profs[var])[:,::-1],axis=1),axis=1)[:,::-1]
                gradErr['xT']+= cond_fun['xT'](x['xT'],'derivative')*np.cumsum((e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0**2*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*dbetadT,axis=2))[:,::-1],axis=1)[:,::-1]
                
#                gradErr['xT']+=e0[var]*(sig_profs[var]*(-2*grad_o) \
#                        +Gain*Const[var]['mult']*forward_profs['Phi']*Tatm0**2*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*dbetadT,axis=2))
                        
            elif 'CombOffline' in var:
                sig = abs_spec['O2off']['sig']
                dsigdT = abs_spec['O2off']['dsigdT']
                Tatm0 = abs_spec['O2off']['Tatm0']
                
#                sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA']['O2off'],forward_profs['T'],Const['base_T'],Const['base_P']) # oxygen absorption cross section
#                sig = sig.reshape(forward_profs['T'].shape)
#                dsigdT = dsigdT.reshape(forward_profs['T'].shape)
                grad_o = cond_fun['xT'](x['xT'],'derivative')*dR*spec.fo2*(dsigdT*forward_profs['nO2'] \
                            +sig*(Cg-1)*Const['base_P'][:,np.newaxis]/(kB*Const['base_T'][:,np.newaxis]**Cg)*forward_profs['T']**(Cg-2))
                
#                Tatm0 = np.exp(-dR*spec.fo2*np.cumsum(sig*forward_profs['nO2'],axis=1))
#                gradErr['xT']+= np.cumsum(grad_o[:,::-1]*np.cumsum((-2*e0[var]*sig_profs[var])[:,::-1],axis=1),axis=1)[:,::-1]
                
                c = -2*e0[var]*sig_profs[var]
                a = grad_o
                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
                
                gradErr['xT']+= cond_fun['xT'](x['xT'],'derivative')*np.cumsum((e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0**2*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*dbetadT,axis=2))[:,::-1],axis=1)[:,::-1]
#                c = -e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0**2
                
                
            elif 'WV' in var:
                if 'Online' in var:
                    spec_def = 'WVon'
                elif 'Offline' in var:
                    spec_def = 'WVoff'
                
                sig = abs_spec[spec_def]['sig']
                dsigdT = abs_spec[spec_def]['dsigdT']
                Tatm0 = abs_spec[spec_def]['Tatm0']
                    
#                sig,dsigdT = spec.calc_pca_T_spectrum_w_deriv(Const['absPCA'][spec_def],forward_profs['T'],Const['base_T'],Const['base_P']) # oxygen absorption cross section
#                sig = sig.reshape(forward_profs['T'].shape)
#                dsigdT = dsigdT.reshape(forward_profs['T'].shape)
                
                c = -2*dR*e0[var]*sig_profs[var]
                a = cond_fun['xT'](x['xT'],'derivative')*dsigdT*forward_profs['nWV']
                gradErr['xT']+= np.cumsum(a[:,::-1]*np.cumsum(c[:,::-1],axis=1),axis=1)[:,::-1]
                
                
                
        if 'xN' in gradErr.keys():
            if not 'WV' in var and 'Online' in var:
                
                sig = abs_spec['O2on']['sig']
                dsigdT = abs_spec['O2on']['dsigdT']
                Tatm = abs_spec['O2on']['Tatm']
                Tatm0 = abs_spec['O2on']['Tatm0']
                
                # water vapor gradient for optical depth    
                grad_o = dR*spec.fo2*sig*cond_fun['xN'](x['xN'],'derivative')[:,:,np.newaxis]
                
                
                
                gradErr['xN']+= -2*grad_o[:,:,i0]*np.cumsum((e0[var]*Gain*Const[var]['mult'] \
                       *forward_profs['phi']*Tatm0**2*Const[var]['Trx'][i0]*(forward_profs['BSR']-1))[:,::-1],axis=1)[:,::-1]
                
                gradErr['xN']+= grad_o[:,:,i0]*np.cumsum((-e0[var]*Gain*Const[var]['mult'] \
                       *forward_profs['phi']*Tatm0*np.nansum(Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta,axis=2))[:,::-1],axis=1)[:,::-1]
                
                gradErr['xN']+= np.nansum(grad_o*np.cumsum((Const[var]['Trx'][np.newaxis,np.newaxis,:]*Tatm*beta)[:,::-1,:],axis=1)[:,::-1,:],axis=2) \
                    *np.cumsum((-e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*Tatm0)[:,::-1],axis=1)[:,::-1]
                
#                # original
#                gradErr['xN']+= -e0[var]*Gain*Const[var]['mult']*forward_profs['Phi']*Tatm0 \
#                        *(2*grad_o[:,:,i0]*Const[var]['Trx'][i0]*Tatm0*(forward_profs['BSR']-1) \
#                        +grad_o[:,:,i0]*np.nansum(Const[var]['Trx']*Tatm*beta,axis=2)+np.nansum(Const[var]['Trx']*Tatm*beta*grad_o,axis=2))
            elif 'CombOffline' in var:
                
                sig = abs_spec['O2off']['sig']
                dsigdT = abs_spec['O2off']['dsigdT']
                Tatm0 = abs_spec['O2off']['Tatm0']
                
                grad_o = -dR*spec.fo2*sig*cond_fun['xN'](x['xN'],'derivative')
                
                gradErr['xN']+= 2*grad_o*grad0[var]
                
            elif 'WV' in var:
                if 'Online' in var:
                    spec_def = 'WVon'
                elif 'Offline' in var:
                    spec_def = 'WVoff'
                
                sig = abs_spec[spec_def]['sig']
                dsigdT = abs_spec[spec_def]['dsigdT']
                Tatm0 = abs_spec[spec_def]['Tatm0']
                
                grad_o = -dR*sig*cond_fun['xN'](x['xN'],'derivative')
                
                gradErr['xN']+= 2*grad_o*grad0[var]
                
        
        if 'xB' in gradErr.keys():
            if not 'WV' in var: 
                if 'Online' in var:
                    gradErr['xB']+= e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*abs_spec['O2on']['Tatm0']**2*Const[var]['Trx'][i0]*cond_fun['xB'](x['xB'],'derivative')
                elif 'Offline' in var:
                    gradErr['xB']+= e0[var]*Gain*Const[var]['mult']*forward_profs['phi']*abs_spec['O2off']['Tatm0']**2*Const[var]['Trx'][i0]*cond_fun['xB'](x['xB'],'derivative')
                    
        if 'xPhi' in gradErr.keys():
            if 'Comb' in var or 'Mol' in var:
                gradErr['xPhi']+=e0[var]*sig_profs[var]*cond_fun['xPhi'](x['xPhi'],'derivative')
        
        if 'xPsi' in gradErr.keys():
            if 'WV' in var:
                gradErr['xPsi']+=e0[var]*sig_profs[var]*cond_fun['xPsi'](x['xPsi'],'derivative')
            
            
    if kconv.size > 1:
        if 'xT' in gradErr.keys():
            gradErr['xT'] = lp.conv2d(gradErr['xT'],kconv,keep_mask=False)
        if 'xB' in gradErr.keys():
            gradErr['xB'] = lp.conv2d(gradErr['xB'],kconv,keep_mask=False)
        if 'xN' in gradErr.keys():
            gradErr['xN'] = lp.conv2d(gradErr['xN'],kconv,keep_mask=False)
        if 'xPhi' in gradErr.keys():
            gradErr['xPhi'] = lp.conv2d(gradErr['xPhi'],kconv,keep_mask=False)
        if 'xPsi' in gradErr.keys():
            gradErr['xPsi'] = lp.conv2d(gradErr['xPsi'],kconv,keep_mask=False)
    
    return gradErr