# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:00:54 2016

@author: rob
"""
import numpy as np
def swiss_roll(ppm=2500, std=2, scale=10, width = 1.0):
  centers = np.array([[0.75, 0.75], [0.75, 1.25],[1.25, 0.75],[ 1.25, 1.25]])
#  centers = np.array([[1.0, 0.75], [0.75, 1.25],[ 1.25, 1.25]])
  centers *= scale
  M = centers.shape[0]
  data = []
  labels = []
  for m in range(M):
    noise = np.random.randn(ppm,2)
#    noise = np.concatenate((np.random.uniform(-1,1,size=(ppm,1)), np.random.uniform(-4,4,size=(ppm,1))),axis=1)
    data.append(width*noise+centers[m,:])
    labels.append(m*np.ones((ppm,1)))
  data = np.concatenate(data,axis=0)
  N = data.shape[0]
  labels = np.concatenate(labels,axis=0)
  data = np.vstack((data[:,0]*np.cos(data[:,0]), data[:,1], data[:,0]*np.sin(data[:,0]))).T
  data += std*np.random.randn(N,3)

  return data, labels