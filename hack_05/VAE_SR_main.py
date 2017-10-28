# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 15:35:57 2016

@author: rob
"""

import sys
sys.path.append('/home/rob/Dropbox/ml_projects/VAE/')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from VAE_SR_model import VAE_SR
from gen_sr import swiss_roll

"""Load the data"""
X_train,y_train = swiss_roll(ppm=500,std=0.1, scale=10, width = 1.0)
X_test,y_test = swiss_roll(ppm=500,std=0.1, scale=10, width = 1.0)

mean = np.mean(X_train,axis=0)
std = np.std(X_train,axis=0)

X_train -= mean
X_train /= std
X_test -= mean
X_test /= std


N,D = X_train.shape
Ntest = X_test.shape[0]
print('Finished loading data')

"""Hyperparameters"""
config = {}
max_iterations = 30000
config['batch_size'] = batch_size = 64
config['learning_rate'] = learning_rate = .005
plot_every = 500      #How often do you want terminal output for the performances
config['num_l'] = num_l = 2  #Number of dimensions in latent space
config['D'] = D

## For a auto-encoder with fully connected layers
config['num_enc1'] = num_enc1 = 30 #Number of neurons in first layer of encoder
config['num_enc2'] = num_enc2 = 20 #Number of neurons in second layer of encoder

#Proclaim the epochs
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))

model = VAE_SR(config)


"""Training time"""

perf_collect = np.zeros((6,int(np.floor(max_iterations /plot_every))))
sess = tf.Session()
sess.run(tf.initialize_all_variables())


step = 0      # Step is a counter for filling the numpy array perf_collect
for i in range(max_iterations):
  batch_ind = np.random.choice(N,batch_size,replace=False)

  if i%plot_every == 0:   #plot_every is how often you want a print to terminal
    #Check training performance
    result = sess.run([model.cost],feed_dict={model.x: X_train[batch_ind]})
    perf_collect[0,step] = cost_train = result[0]

    #Check testperformance
    batch_ind_test = np.random.choice(Ntest,batch_size,replace=False)
    result = sess.run([model.cost, model.loss_kld, model.loss_rec],feed_dict={model.x: X_test[batch_ind_test]})
    perf_collect[1,step] = cost_test = result[0]

    #Write information to TensorBoard
    print('At %5.0f/%5.0f, cost is %6.3f(train) %6.3f(val)  (%7.3f, %7.3f)'%(i,max_iterations,cost_train,cost_test, np.mean(result[1]), np.mean(result[2])))
    step +=1
  sess.run([model.train_step],feed_dict={model.x:X_train[batch_ind]})




"""Plot the evolution of cost"""
plt.figure()
plt.plot(perf_collect[0,:],label='train cost')
plt.plot(perf_collect[1,:],label='val cost')
plt.legend()
plt.title('Cost evolution')
plt.show()

lat_scatter = True   #Do you want a scatter plot of the latent space?
canvas_vis = False   #Do you want a canvas visualization of the latent space

if num_l == 2 and lat_scatter:
  plt.figure()
  z_lat = sess.run(model.z_mu,feed_dict = {model.x:X_test})
  plt.scatter(z_lat[:,0],z_lat[:,1],c=y_test)
  plt.colorbar()
  plt.savefig('swiss_roll_VAE.eps',format='eps',dpi=1500)

if num_l == 2 and canvas_vis:
  plt.figure()
  Ntiles = 18
  z1 = np.linspace(-3.0,3.0,Ntiles)
  z2 = np.linspace(-3.0,3.0,Ntiles)
  Z1,Z2 = np.meshgrid(z1,z2)
  z_feed = np.vstack((Z1.flatten(),Z2.flatten())).T
  x_rec = sess.run(model.rec,feed_dict={model.z:z_feed})
  canvas_stack = np.reshape(x_rec,(Ntiles,Ntiles,28,28))
  length = Ntiles*28
  canvas = np.zeros((length,length))
  for ix,xstart in enumerate(np.arange(0,length,28)):
    for iy,ystart in enumerate(np.arange(0,length,28)):
      canvas[xstart:xstart+28,ystart:ystart+28] = canvas_stack[ix,iy,:,:].T
  plt.imshow(canvas)
  plt.colorbar()