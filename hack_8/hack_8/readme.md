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