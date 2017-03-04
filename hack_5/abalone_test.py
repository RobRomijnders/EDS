# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:40:25 2016

@author: rob
"""

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from scipy.stats import zscore
import os.path
import cPickle
import pandas as pd

df = pd.read_csv('abalone.data')
df['M'] = df['M'].map({'M':0.0, 'F':1.0, 'I':2.0})

X = np.array(df.as_matrix())

data = X[:,1:-1]
labels = X[:,-1]

data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)+1e-12


## Save data
N = data.shape[0]
shuffle = np.random.permutation(N)

np.save('abalone_dataset.npy',data[shuffle])
np.save('abalone_labels.npy',labels[shuffle])

##PCA
PCA_model = TruncatedSVD(n_components=2).fit(data)
wine_reduced = PCA_model.transform(data)
f, axarr = plt.subplots(2, 2)
axarr[0, 0].scatter(wine_reduced[:,0],wine_reduced[:,1],c=labels,marker='*',linewidths = 0)
axarr[0, 0].set_title('PCA on Abalone')
axarr[0, 1].scatter(wine_reduced[:,0],wine_reduced[:,1],c = 'k',marker='*',linewidths = 0)
axarr[0, 1].set_title('PCA on Abalone - uncolored')

##tSNE
tSNE_model = TSNE(verbose=2)#, early_exaggeration=6, n_iter = 3000,min_grad_norm=1e-12, n_iter_without_progress=50, perplexity=70)
if True: #not os.path.isfile("wine_tSNE.p"): 
  wine_tSNE = tSNE_model.fit_transform(data)
  cPickle.dump( wine_tSNE, open( "abalone_tSNE.p", "wb" ) )
else:
  wine_tSNE = cPickle.load( open( "abalone_tSNE.p", "rb" ) )

axarr[1,0].scatter(wine_tSNE[:,0],wine_tSNE[:,1],c=labels,marker='*',linewidths = 0)
axarr[1,0].set_title('tSNE on Abalone')
axarr[1,1].scatter(wine_tSNE[:,0],wine_tSNE[:,1],c='k',marker='*',linewidths = 0)
axarr[1,1].set_title('tSNE on Abalone - uncolored')