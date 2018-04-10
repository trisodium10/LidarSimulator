# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:00:11 2017

@author: mhayman
"""

import numpy as np
import scipy.io

"""
Generates the Rayleigh-Brillioun data file for python scripts.
The PCA is performed in Matlab using RB_PCA.m

"""

#### Read Rayleigh Brillouin PCA Matrix Data
MatFile = '/Users/mhayman/Documents/MATLAB/RB_PCA_Params.mat'
matdata = scipy.io.loadmat(MatFile)
M = matdata['M']
Mavg = matdata['Mavg']
x = matdata['x1d'].T
dM = matdata['dM']

np.savez('RB_PCA_Params',M=M,Mavg=Mavg,x=x,dM=dM)