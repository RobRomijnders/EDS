# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 20:36:44 2016

@author: rob
"""

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from scipy.stats import zscore
import os.path
import cPickle

data_red = np.loadtxt('winequality-red.csv', delimiter = ';', skiprows=1)
data_white = np.loadtxt('winequality-white.csv', delimiter = ';', skiprows=1)
Nwhite = data_white.shape[0]
Nred = data_red.shape[0]

if False:
  labels_white = data_white[:,-1]
  data_white = data_white[:,:-1]
else:
  dataset = np.concatenate((data_red, data_white),axis=0)
  data_white = dataset[:,:-1]
  labels_white = dataset[:,-1]
  labels_white = np.concatenate((np.zeros((Nred,1)), np.ones((Nwhite,1))),axis=0)

N = data_white.shape[0]
shuffle = np.random.permutation(N)

np.save('wine_dataset.npy',data_white[shuffle])
np.save('wine_labels.npy',labels_white[shuffle])


data_white -= np.mean(data_white, axis=0)
data_white /= np.std(data_white, axis=0)+1e-12


##PCA
PCA_model = TruncatedSVD(n_components=2).fit(data_white)
wine_reduced = PCA_model.transform(data_white)
f, axarr = plt.subplots(2, 2)
axarr[0, 0].scatter(wine_reduced[:,0],wine_reduced[:,1],c=labels_white,marker='*',linewidths = 0)
axarr[0, 0].set_title('PCA on Wine')
axarr[0, 1].scatter(wine_reduced[:,0],wine_reduced[:,1],c = 'k',marker='*',linewidths = 0)
axarr[0, 1].set_title('PCA on Wine - uncolored')

##tSNE
tSNE_model = TSNE(verbose=2, early_exaggeration=6, n_iter = 3000,min_grad_norm=1e-12, n_iter_without_progress=50, perplexity=70)
if True: #not os.path.isfile("wine_tSNE.p"): 
  wine_tSNE = tSNE_model.fit_transform(data_white)
  cPickle.dump( wine_tSNE, open( "wine_tSNE.p", "wb" ) )
else:
  wine_tSNE = cPickle.load( open( "wine_tSNE.p", "rb" ) )

axarr[1,0].scatter(wine_tSNE[:,0],wine_tSNE[:,1],c=labels_white,marker='*',linewidths = 0)
axarr[1,0].set_title('tSNE on Wine')
axarr[1,1].scatter(wine_tSNE[:,0],wine_tSNE[:,1],c='k',marker='*',linewidths = 0)
axarr[1,1].set_title('tSNE on Wine - uncolored')
