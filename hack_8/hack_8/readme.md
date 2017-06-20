# Hackathon 8

  * Subject: Introduction to Neural Networks with Tensorflow
  * Location: Glaspaviljoen, Eindhoven. (Location has a bar, please bring cash)

# Preparation
__Please work through this preparation before you attend. So we can make fast progress during the meetup__

## Installing Tensorflow
Walk through [these](https://www.tensorflow.org/install/) steps to install Tensorflow

To check your installation, start a Python console and run:
```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```
Which should print
```python
>>> Hello, TensorFlow!
```

## Know the building blocks
Tensorflow implements a computation graph which represents a neural network or some other model.
We'll start the meetup explaining what are placeholders (`tf.placeholder()`), nodes and variables (`tf.Variable()`). You will
learn more if you've read about this as a preparation. [Here's an introduction (you may skip the section on tf.contrib).](https://www.tensorflow.org/get_started/get_started)

After reading on these topics, you should be able to answer:

  * Why write a node like `a = tf.constant(3.0)`?
  * If `b = tf.constant(4.0)`, what is the result of `print(a+b)`? Why doesn't this print `7.0`?
  * What is the difference between `tf.placeholder()` and `tf.Variable()`?
  * Why do we run `sess.run(tf.global_variables_initializer())` before training any neural network with Tensorflow?
  
  
If you have any comments or questions on the preparation, please file an issue on github or email us at [romijndersrob@gmail.com](mailto:romijndersrob@gmail.com)

# Meetup

Questionnaire: [Fill in the questionnaire here](https://docs.google.com/forms/d/e/1FAIpQLSfG3qudK691ZxcaKncwiNkj2Ncn8PnXl0-2aug-Bz78DgnIIg/viewform?usp=sf_link)

Useful links

  * [Visualizing a multi layer perceptron](http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html)
  * [Getting started with Tensorflow](https://www.tensorflow.org/get_started/)
  * [Tensorflow Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.61950&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)


Reading for the next meetup

  * [The unreasonable effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  * [CS231n: convolutional neural networks](https://www.youtube.com/watch?v=NfnWJUyUJYU&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)
  * [Udacity course on Deep Learning](https://www.udacity.com/course/deep-learning--ud730)
