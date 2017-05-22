from hack_8.util.dataloader import read_data_sets
import tensorflow as tf


#Load the data and run once to see if it works
MNIST = read_data_sets('../data/', norm=True)
X_batch, y_batch = MNIST.next_flat_batch(32,'train')

#Make the placeholders
X_ph = tf.placeholder(tf.float32, [None,784],name="X_placeholder")
y_ph = tf.placeholder(tf.int64, [None,],name="y_placeholder")

#Define the network
#Assignment 3: extend the computation to a neural network
W = tf.get_variable('weight', [784, 10])
b = tf.get_variable('bias',[10,])

a = tf.nn.xw_plus_b(X_ph, W, b)
logits = tf.nn.softmax(a)

#Calculate the loss
loss_node = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph, logits=logits)
loss_batch = tf.reduce_mean(loss_node)

#Assignment 1: calculate the accuracy here

#Make an SGD step
step = tf.train.AdamOptimizer(0.0005).minimize(loss_batch)

#Start a Tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for k in range(5000):
    X_batch, y_batch = MNIST.next_flat_batch()
    loss, _ = sess.run([loss_batch, step],{X_ph:X_batch,y_ph:y_batch})
    #Assignment2: print the loss on the validation set (once every 100 steps)
    if k%100 == 0: print(loss)






