import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

def minimize(loss_batch, learningrate=0.0005):
    """The following plots for every trainable variable
      - Histogram of the entries of the Tensor
      - Histogram of the gradient over the Tensor
      - Histogram of the gradient-norm over the Tensor"""
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss_batch, tvars)
    gradients = zip(grads, tvars)
    step = tf.train.AdamOptimizer(learningrate).apply_gradients(gradients)

    for gradient, variable in zip(grads, tvars):
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient

        h1 = tf.summary.histogram(variable.name, variable)
        h2 = tf.summary.histogram(variable.name + "/gradients", grad_values)
        h3 = tf.summary.histogram(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
    return step