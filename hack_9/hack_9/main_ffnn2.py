from util.dataloader import read_data_sets
from util.util_func import minimize
import tensorflow as tf
from time import gmtime, strftime


#Start a Tensorflow session
sess = tf.Session()

#Load the data and run once to see if it works
MNIST = read_data_sets('../data/', norm=True)
X_batch, y_batch = MNIST.next_batch(32,'train')

#Make the placeholders
X_ph = tf.placeholder(tf.float32, [None,28,28,1],name="X_placeholder")
y_ph = tf.placeholder(tf.int64, [None,],name="y_placeholder")

# Assignment 1: Replace the first "tf.nn.xw_plus_b()" functions with a convolution
# Tip: at some point you must delete the next line
X = tf.reshape(X_ph,[-1,784])

#Define the network
W1 = tf.get_variable('weight1', [784, 100])
b1 = tf.get_variable('bias1',[100,])
W2 = tf.get_variable('weight2',[100,10])
b2 = tf.get_variable('bias2',[10,])


a1 = tf.nn.xw_plus_b(X, W1, b1)
h1 = tf.nn.tanh(a1)
a2 = tf.nn.xw_plus_b(h1,W2,b2)
logits = tf.nn.softmax(a2)

#Calculate the loss
loss_node = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph, logits=logits)
loss_batch = tf.reduce_mean(loss_node)

#Calculate the accuracy
pred = tf.argmax(logits,1)
acc_node = tf.reduce_mean(tf.cast(tf.equal(y_ph,pred),tf.float32))

#Make an SGD step
step = minimize(loss_batch, learningrate=0.0005)

#Make node for Tensorboard
tensorboard = tf.summary.merge_all()
writer = tf.summary.FileWriter('/tmp/tensorboard/'+strftime("%Y-%m-%d_%H-%M-%S", gmtime()),sess.graph)

#Initialize the variables
sess.run(tf.global_variables_initializer())

for k in range(5000):
    X_batch, y_batch = MNIST.next_batch()
    loss, _, acc = sess.run([loss_batch, step, acc_node],{X_ph:X_batch,y_ph:y_batch})
    if k%100 == 0:
        X_val, y_val = MNIST.next_batch(dataset='val')
        acc_val, log_string = sess.run([acc_node, tensorboard], feed_dict={X_ph:X_val, y_ph:y_val})
        writer.add_summary(log_string, k)
        print('At step %5i, loss %5.3f and accuracy TRAIN %5.3f VAL %5.3f'%(k,loss,acc, acc_val))






