# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:48:48 2016

@author: rob

Download MNIST from:
http://deeplearning.net/tutorial/gettingstarted.html

Download Breast Cancer dataset from
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Download Swiss roll from
http://isomap.stanford.edu/datasets.html
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

#Swiss roll
SR_data,SR_labels = swiss_roll(ppm=500,std=0.5, scale=10, width = 1.0)

"""Check swiss roll"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(SR_data[:,0],SR_data[:,1],SR_data[:,2],c=SR_labels)

"""PCA"""
##MNIST
PCA_model = TruncatedSVD(n_components=2).fit(MNIST_train[0])
MNIST_reduced = PCA_model.transform(MNIST_valid[0])
plt.figure()
plt.scatter(MNIST_reduced[:,0],MNIST_reduced[:,1],c=MNIST_valid[1],marker='*',linewidths = 0)
plt.title('PCA on MNIST')

##WDBC
PCA_model_wdbc = TruncatedSVD(n_components=2).fit(WDBC_data)
WDBC_reduced = PCA_model_wdbc.transform(WDBC_data)
plt.figure()
plt.scatter(WDBC_reduced[:,0],WDBC_reduced[:,1],c=WDBC_labels,marker='*',linewidths = 0)
plt.title('PCA on WDBC')

##Swiss Roll
PCA_model_sr = TruncatedSVD(n_components=2).fit(SR_data)
SR_reduced = PCA_model_sr.transform(SR_data)
plt.figure()
plt.scatter(SR_reduced[:,0],SR_reduced[:,1],c=SR_labels,marker='*',linewidths = 0)
plt.title('PCA on SR')

"""tSNE"""

##MNIST
tSNE_model = TSNE(verbose=1)
if not os.path.isfile("MNIST_tSNE.p"): 
  MNIST_tSNE = tSNE_model.fit_transform(MNIST_valid[0])
  cPickle.dump( MNIST_tSNE, open( "MNIST_tSNE.p", "wb" ) )
else:
  MNIST_tSNE = cPickle.load( open( "MNIST_tSNE.p", "rb" ) )

plt.figure()
plt.scatter(MNIST_tSNE[:,0],MNIST_tSNE[:,1],c=MNIST_valid[1],marker='*',linewidths = 0)
plt.title('tSNE on MNIST')

##WDBC
tSNE_model_wdbc = TSNE(verbose=1)
if not os.path.isfile("WDBC_tSNE.p"): 
  WDBC_tSNE = tSNE_model_wdbc.fit_transform(WDBC_data)
  cPickle.dump( WDBC_tSNE, open( "WDBC_tSNE.p", "wb" ) )
else:
  WDBC_tSNE = cPickle.load( open( "WDBC_tSNE.p", "rb" ) )

plt.figure()
plt.scatter(WDBC_tSNE[:,0],WDBC_tSNE[:,1],c=WDBC_labels,s=50,marker='*',linewidths = 0)
plt.title('tSNE on WDBC')

##Swiss Roll
tSNE_model_SR = TSNE(verbose=1,perplexity=50,min_grad_norm=1E-12,n_iter=3000)
SR_tSNE = tSNE_model_SR.fit_transform(SR_data)

plt.figure()
plt.scatter(SR_tSNE[:,0],SR_tSNE[:,1],c=SR_labels,s=50,marker='*',linewidths = 0)
plt.title('tSNE on WDBC')
