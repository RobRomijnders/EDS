# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 19:11:08 2016

@author: rob
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
def tf_1d_normal(x,mu,s):
  """ 1D Normal distribution for tensorflow
  input
  - signal
  - mean
  - sigma 
  """
  norm = tf.sub(x, mu)
  z = tf.square(tf.div(norm, s))
  result = tf.exp(tf.div(-z,2))
  denom = 2.0*np.pi*s
  px = tf.div(result, denom)  #probability in x dimension
  return px

class VAE_SR():
  def __init__(self,config):
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_enc1 = config['num_enc1']
    num_enc2 = config['num_enc2']
    num_l = config['num_l']
    D = config['D']

    with tf.name_scope("Placeholders") as scope:
      self.x = tf.placeholder("float", shape=[None, D], name = 'Input_data')

    with tf.name_scope("Encoding_network") as scope:
      #Layer 1
      W1e = tf.get_variable("W1e", [D,num_enc1])
      b1e = tf.Variable(tf.constant(0.1,shape=[num_enc1],dtype=tf.float32))
      h1e = tf.nn.relu(tf.nn.xw_plus_b(self.x,W1e,b1e))

      #Layer 1
      W2e = tf.get_variable("W2e", [num_enc1,num_enc2])
      b2e = tf.Variable(tf.constant(0.1,shape=[num_enc2],dtype=tf.float32))
      h2e = tf.nn.relu(tf.nn.xw_plus_b(h1e,W2e,b2e))

      #layer for mean of z
      W_mu = tf.get_variable("W_mu", [num_enc2,num_l])
      b_mu = tf.Variable(tf.constant(0.1,shape=[num_l],dtype=tf.float32))
      self.z_mu = tf.nn.xw_plus_b(h2e,W_mu,b_mu)  #mu, mean, of latent space

      #layer for sigma of z
      W_sig = tf.get_variable("W_sig", [num_enc2,num_l])
      b_sig = tf.Variable(tf.constant(0.1,shape=[num_l],dtype=tf.float32))
      z_sig_log_sq = tf.nn.xw_plus_b(h2e,W_sig,b_sig)  #sigma of latent space, in log-scale and squared.
      # This log_sq will save computation later on. log(sig^2) is a real number, so no sigmoid is necessary

    with tf.name_scope("Latent_space") as scope:
      self.eps = tf.random_normal(tf.shape(self.z_mu),0,1,dtype=tf.float32)
      self.z = self.z_mu + tf.mul(tf.sqrt(tf.exp(z_sig_log_sq)),self.eps)

    with tf.name_scope("Decoding_network") as scope:
      #Layer 1
      W1d = tf.get_variable("W1d", [num_l,num_enc2])
      b1d = tf.Variable(tf.constant(0.1,shape=[num_enc2],dtype=tf.float32))
      h1d = tf.nn.relu(tf.nn.xw_plus_b(self.z,W1d,b1d))

      #Layer 1
      W2d = tf.get_variable("W2d", [num_enc2,num_enc1])
      b2d = tf.Variable(tf.constant(0.01,shape=[num_enc1],dtype=tf.float32))
      h2d = tf.nn.relu(tf.nn.xw_plus_b(h1d,W2d,b2d))

      #Layer for reconstruction
      W_rec = tf.get_variable("W_rec", [num_enc1,2*D])
      b_rec = tf.Variable(tf.constant(0.5,shape=[2*D],dtype=tf.float32))
      self.h_rec = tf.nn.xw_plus_b(h2d,W_rec,b_rec)  #Reconstruction.
      self.mu_rec, self.sigma_rec = tf.split(1, 2, self.h_rec)  #First D are mu, second D are sigma
      self.sigma_rec = tf.exp(self.sigma_rec)
      self.p_rec = tf_1d_normal(self.x, self.mu_rec, self.sigma_rec)

    with tf.name_scope("Loss_calculation") as scope:
      #See equation (10) of https://arxiv.org/abs/1312.6114
      self.loss_rec = tf.reduce_sum(-tf.log(tf.maximum(self.p_rec, 1e-20)), 1)  #recovery loss expressed as negative log-likelihood
      self.loss_kld = -0.5*tf.reduce_sum((1+z_sig_log_sq-tf.square(self.z_mu)-tf.exp(z_sig_log_sq)),1)   #KL divergence

      self.cost = tf.reduce_mean(self.loss_rec + self.loss_kld)

    with tf.name_scope("Optimization") as scope:
      tvars = tf.trainable_variables()
      global_step = tf.Variable(0,trainable=False)
      lr = tf.train.exponential_decay(learning_rate,global_step,30000,0.7,staircase=True)
      #We clip the gradients to prevent explosion
      grads = tf.gradients(self.cost, tvars)
      optimizer = tf.train.AdamOptimizer(lr)
      gradients = zip(grads, tvars)
      self.train_step = optimizer.apply_gradients(gradients,global_step=global_step)
    print('Finished computation graph')


