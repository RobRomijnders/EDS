# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:48:48 2016

@author: rob

Download MNIST from:
http://deeplearning.net/tutorial/gettingstarted.html

Download Breast Cancer dataset from
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

"""

import cPickle, gzip
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from scipy.stats import zscore
import os.path
from gen_sr import swiss_roll


# LOAD MNIST
f = gzip.open('mnist.pkl.gz', 'rb')
MNIST_train, MNIST_valid, test_set = cPickle.load(f)
f.close()

# Load Breast cancer dataset
WDBC_labels = (np.genfromtxt('wdbc.data',delimiter=',', usecols = 1,dtype=str) == 'M').astype(np.int)
WDBC_data = zscore(np.genfromtxt('wdbc.data',delimiter=',')[:,2:])

# Load Swiss roll
SR_data,SR_labels = swiss_roll(ppm=1000,std=0.3, scale=10, width = 1.0)


##Check swiss roll
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(SR_data[:,0],SR_data[:,1],SR_data[:,2],c=SR_labels)
"""Plot code"""
f1, ax1 = plt.subplots(2, 2)
f2, ax2 = plt.subplots(2, 2)
f3, ax3 = plt.subplots(2, 2)

"""PCA"""
##MNIST
PCA_model = TruncatedSVD(n_components=2).fit(MNIST_train[0])
MNIST_reduced = PCA_model.transform(MNIST_valid[0])
ax1[0,0].scatter(MNIST_reduced[:,0],MNIST_reduced[:,1],c=MNIST_valid[1],marker='*',linewidths = 0)
ax1[0,1].scatter(MNIST_reduced[:,0],MNIST_reduced[:,1],c='k',marker='*',linewidths = 0)
ax1[0,0].set_title('PCA on MNIST')
ax1[0,1].set_title('PCA on MNIST')


##WDBC
PCA_model_wdbc = TruncatedSVD(n_components=2).fit(WDBC_data)
WDBC_reduced = PCA_model_wdbc.transform(WDBC_data)
ax2[0,0].scatter(WDBC_reduced[:,0],WDBC_reduced[:,1],c=WDBC_labels,marker='*',linewidths = 0)
ax2[0,1].scatter(WDBC_reduced[:,0],WDBC_reduced[:,1],c='k',marker='*',linewidths = 0)
ax2[0,0].set_title('PCA on WDBC')
ax2[0,1].set_title('PCA on WDBC')


##Swiss Roll
PCA_model_sr = TruncatedSVD(n_components=2).fit(SR_data)
SR_reduced = PCA_model_sr.transform(SR_data)
ax3[0,0].scatter(SR_reduced[:,0],SR_reduced[:,1],c=SR_labels,marker='*',linewidths = 0)
ax3[0,1].scatter(SR_reduced[:,0],SR_reduced[:,1],c='k',marker='*',linewidths = 0)
ax3[0,0].set_title('PCA on SR')
ax3[0,1].set_title('PCA on SR')



"""tSNE"""
##MNIST
tSNE_model = TSNE(verbose=2, perplexity=30,min_grad_norm=1E-12,n_iter=3000)
if not os.path.isfile("MNIST_tSNE.p"): 
  MNIST_tSNE = tSNE_model.fit_transform(MNIST_valid[0])
  cPickle.dump( MNIST_tSNE, open( "MNIST_tSNE.p", "wb" ) )
else:
  MNIST_tSNE = cPickle.load( open( "MNIST_tSNE.p", "rb" ) )

ax1[1,0].scatter(MNIST_tSNE[:,0],MNIST_tSNE[:,1],c=MNIST_valid[1],marker='*',linewidths = 0)
ax1[1,1].scatter(MNIST_tSNE[:,0],MNIST_tSNE[:,1],c='k',marker='*',linewidths = 0)
ax1[1,0].set_title('tSNE on MNIST')
ax1[1,1].set_title('tSNE on MNIST')


##WDBC
tSNE_model_wdbc = TSNE(verbose=1)
if not os.path.isfile("WDBC_tSNE.p"): 
  WDBC_tSNE = tSNE_model_wdbc.fit_transform(WDBC_data)
  cPickle.dump( WDBC_tSNE, open( "WDBC_tSNE.p", "wb" ) )
else:
  WDBC_tSNE = cPickle.load( open( "WDBC_tSNE.p", "rb" ) )

ax2[1,0].scatter(WDBC_tSNE[:,0],WDBC_tSNE[:,1],c=WDBC_labels,s=50,marker='*',linewidths = 0)
ax2[1,1].scatter(WDBC_tSNE[:,0],WDBC_tSNE[:,1],c='k',s=50,marker='*',linewidths = 0)
ax2[1,0].set_title('tSNE on WDBC')
ax2[1,1].set_title('tSNE on WDBC')


##Swiss Roll
tSNE_model_SR = TSNE(verbose=2, early_exaggeration=5, n_iter = 3000,min_grad_norm=1e-12, n_iter_without_progress=100, perplexity=170)
SR_tSNE = tSNE_model_SR.fit_transform(SR_data)

ax3[1,0].scatter(SR_tSNE[:,0],SR_tSNE[:,1],c=SR_labels,s=50,marker='*',linewidths = 0)
ax3[1,1].scatter(SR_tSNE[:,0],SR_tSNE[:,1],c='k',s=50,marker='*',linewidths = 0)
ax3[1,0].set_title('tSNE on Swiss Roll')
ax3[1,1].set_title('tSNE on Swiss Roll')

