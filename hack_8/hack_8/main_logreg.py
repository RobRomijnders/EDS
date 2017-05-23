from util.dataloader import read_data_sets
import tensorflow as tf


#Load the data and run once to see if it works
MNIST = read_data_sets('../data/', norm=True)
X_batch, y_batch = MNIST.next_flat_batch(32,'train')

#Make the placeholders
X_ph = tf.placeholder(tf.float32, [None,784],name="X_placeholder")
y_ph = tf.placeholder(tf.int64, [None,],name="y_placeholder")

#Define the network
#Assignment 3: extend the computation to a neural network
W = tf.get_variable('weight', [784, 100])
b = tf.get_variable('bias',[100,])
W2 = tf.get_variable('weight1',[100,10])
b2 = tf.get_variable('bias1',[10,])


a1 = tf.nn.xw_plus_b(X_ph, W, b)
h1 = tf.nn.tanh(a1)
a2 = tf.nn.xw_plus_b(h1,W2,b2)
logits = tf.nn.softmax(a2)

#Calculate the loss
loss_node = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph, logits=logits)
loss_batch = tf.reduce_mean(loss_node)

#Assignment 1: calculate the accuracy here
pred = tf.argmax(logits,1)
acc_node = tf.reduce_mean(tf.cast(tf.equal(y_ph,pred),tf.float32))

#Make an SGD step
step = tf.train.AdamOptimizer(0.0005).minimize(loss_batch)

#Start a Tensorflow session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for k in range(5000):
    X_batch, y_batch = MNIST.next_flat_batch()
    loss, _, acc = sess.run([loss_batch, step, acc_node],{X_ph:X_batch,y_ph:y_batch})
    #Assignment2: print the loss on the validation set (once every 100 steps)
    if k%100 == 0:
        X_val, y_val = MNIST.next_flat_batch(dataset='val')
        acc_val = sess.run(acc_node, feed_dict={X_ph:X_val, y_ph:y_val})
        print('At step %5i, loss %5.3f and accuracy TRAIN %5.3f VAL %5.3f'%(k,loss,acc, acc_val))






