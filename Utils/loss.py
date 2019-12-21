import numpy as np
import tensorflow as tf

class Loss():
  '''
  over-riding loss class. Based off arXiv:1702.05659
  '''
  def __init__(self, logits,
               labels,
               lss_fn,
               vocab_size = None,
               label_smoothing = 1.0,
               activation = tf.nn.softmax,
               huber_delta = 1.0):
    '''
    logits - tensor, dtype = tf.float32 shape = [..., vocab_size]
    labels - tensor, dtype = tf.float32 shape = [..., vocab_size], or dtype = tf.int32 shape = [...]
    lss_fn - the loss function used: Potential loss functions include:
      * l1_loss
      * l2_loss
      * l1_expectation_loss
      * l2_regularized_loss
      * chebyshev_loss
      * hinge_loss
      * squared_hinge_loss
      * cubed_hinge_loss
      * log_loss
      * squared_log_loss
      * tanimoto_loss
      * cauchy_schwarz_div
      * huber_loss
      * cosine_distance
      * sparse_softmax_cross_entropy_with_logits (most common tensorflow loss function)
    vocab_size - if not specified, calculated as the final dimension of the logits tensor
    label_smoothing - value for label_smoothing. Initial value specifies there will be no label_smoothing
    activation - the activation function that the logits are sent through before the final loss function is calculated
    huber_delta - a hyperparameter used for huber_loss
    '''
    self.lss_fn = lss_fn
    self.label_smoothing = label_smoothing
    self.activation = activation
    self.huber_delta = huber_delta
    if lss_fn == 'sparse_softmax_cross_entropy_with_logits':
      if label_smoothing != 1.0:
        print('sparse_softmax_cross_entropy_with_logits cannot be used with label smoothing')
      if len(labels.shape.as_list()) == len(logits.shape.as_list()):
        labels = tf.argmax(labels,
                           axis = -1)
      self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels,
                                                                 logits = logits)
    elif lss_fn == 'softmax_cross_entropy_with_logits':
      if label_smoothing != 1.0:
        print('softmax_cross_entropy_with_logits cannot be used with label smoothing')
      self.loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels,
                                                          logits = logits)
    else:
      if vocab_size == None:
        vocab_size = logits.shape.as_list()[-1]
      if len(logits.shape.as_list()) - 1 == len(labels.shape.as_list()):
        labels = tf.one_hot(labels,
                            vocab_size,
                            dtype = tf.float32)
      else:
        assert labels.dtype == logits.dtype
      if label_smoothing != 1.0:
        incorrect_label = (1.0 - label_smoothing) / (vocab_size - 1)
        labels *= (label_smoothing - incorrect_label)
        labels += incorrect_label
      if lss_fn == 'l1_loss':
        self.loss = tf.norm(logits - labels,
                            ord = 1,
                            axis = -1)
      elif lss_fn == 'l2_loss':
        self.loss = tf.square(tf.norm(logits - labels,
                                      ord = 2,
                                      axis = -1))
      elif lss_fn == 'l1_expectation_loss':
        self.loss = tf.norm(activation(logits) - labels,
                            ord = 1,
                            axis = -1)
      elif lss_fn == 'l2_regularized_loss':
        self.loss = tf.square(tf.norm(activation(logits) - labels,
                                      ord = 2,
                                      axis = -1))
      elif lss_fn == 'chebyshev_loss':
        self.loss = tf.reduce_max(tf.abs(logits - labels),
                                  axis = -1)
      elif lss_fn == 'hinge_loss':
        labels = labels*2 - 1
        self.loss = tf.nn.relu(0.5 - tf.multiply(logits,
                                                 labels))
      elif lss_fn == 'squared_hinge_loss':
        labels = labels*2 - 1
        self.loss = tf.square(tf.nn.relu(0.5 - tf.multiply(logits,
                                                           labels)))
      elif lss_fn == 'cubed_hinge_loss':
        labels = labels*2 - 1
        self.loss = tf.pow(tf.nn.relu(0.5 - tf.multiply(logits,
                                                        labels)), 3)
      elif lss_fn == 'log_loss':
        self.loss = -labels * self.log(activation(logits)) - (1 - labels) * self.log(1 - activation(logits))
      elif lss_fn == 'squared_log_loss':
        self.loss = tf.square(-labels * self.log(activation(logits)) - (1 - labels) * self.log(1 - activation(logits)))
      elif lss_fn == 'tanimoto_loss':
        num = -tf.reduce_sum(tf.multiply(activation(logits),
                                         labels))
        den = tf.square(tf.norm(activation(logits),
                                ord = 2,
                                axis = -1)) + tf.square(tf.norm(labels,
                                                                ord = 2,
                                                                axis = -1)) + num
        self.loss = num/den
      elif lss_fn == 'cauchy_schwarz_div':
        num = tf.reduce_sum(tf.multiply(activation(logits),
                                        labels),
                            axis = -1)
        den = tf.multiply(tf.norm(activation(logits),
                                  ord = 2,
                                  axis = -1), tf.norm(labels,
                                                      ord = 2,
                                                      axis = -1))
        self.loss = - self.log(num/den)
      elif lss_fn == 'pseudo_huber_loss' or lss_fn == 'huber_loss':
        self.loss = huber_delta**2 * (tf.sqrt(1 + tf.square((logits - labels)/huber_delta)) - 1)
      elif lss_fn == 'cosine_distance':
        num = tf.reduce_sum(tf.multiply(logits,
                                        labels), 
                            axis = -1)
        den = tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(logits),
                                                axis = -1)), 
                          tf.sqrt(tf.reduce_sum(tf.square(labels),
                                                axis = -1)))
        self.loss = -num/den
        
  def log(self, x):
    x = tf.maximum(x,
                   1e-8)
    return tf.log(x)
  
  def l2_loss(self, trainable_weights):
    '''
    given the trainable_weights of the neural network, this returns the auxiallary loss for weight decay
    '''
    l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    return l2_regularizer
  
if __name__ == '__main__':
  losses = ['l1_loss', 'l2_loss', 'l1_expectation_loss', 'l2_regularized_loss', 'chebyshev_loss', 'hinge_loss', 'squared_hinge_loss', 'cubed_hinge_loss', 'log_loss', 
            'squared_log_loss', 'tanimoto_loss', 'cauchy_schwarz_div', 'huber_loss', 'cosine_distance', 'sparse_softmax_cross_entropy_with_logits']
  
  logits = tf.placeholder(tf.float32,
                          shape = [32, 100])
  labels = tf.placeholder(tf.int32,
                          [32])
  for loss in losses:
    loss_cl = Loss(logits,
                   labels,
                   loss,
                   vocab_size = 100)
    print(loss_cl.loss)
    print(loss)