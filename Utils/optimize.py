import numpy as np
import tensorflow as tf

class Optimizer():
  '''
  An over-riding Optimizer class that can be called across all models
  '''
  def __init__(self, arg,
               loss,
               learning_rate = None,
               optimizer = None,
               distribution_strategy = None):
    '''
    arg - the argument class; see below
    loss - the loss function calculated from a neural network
    learning_rate - the learning rate of the optimizer
    optimizer - the optimizer class. Initial class is the AdamOptimizer, but can be replaced with any optimizer chosen
    
    when calculating the gradients of the neural network, the following calculations can be made:
     - random gaussian can be added to the gradients
     - the gradients can be clipped by norm
     - the gradients can be clipped by values
    '''
    self.arg = arg
    if optimizer:
      self.optimizer = optimizer
    else:
      self.optimizer = tf.train.AdamOptimizer(learning_rate)
    self.grads_vars = self.optimizer.compute_gradients(loss)
    if self.arg.grad_noise:
      self.grads_vars = [(self.add_gradient_noise(g, 
                                                  stddev = self.arg.stddev_noise), 
                          v) for g, v in self.grads_vars]
    if self.arg.clip_norm:
      self.grads_vars = [(self.clip_by_norm(g,
                                            self.arg.norm_value),
                          v) for g, v in self.grads_vars]
    if self.arg.clip_value:
      self.grads_vars = [(self.clip_by_value(g,
                                             self.arg.clip_value_min,
                                             self.arg.clip_value_max),
                          v) for g, v in self.grads_vars]
    self.train_op = self.optimizer.apply_gradients(self.grads_vars)
    
  def add_gradient_noise(self, gradient,
                         stddev):
    if gradient is None:
      return None
    return gradient + tf.random_normal(tf.shape(gradient),
                                       stddev = stddev)
    
  def clip_by_norm(self, grad,
                   norm_value):
    if grad == None:
      return None
    else:
      return tf.clip_by_norm(grad,
                             norm_value)
    
  def clip_by_value(self, grad,
                    min_value,
                    max_value):
    if grad == None:
      return None
    else:
      return tf.clip_by_value(grad,
                              min_value,
                              max_value)
    
  def get_accuracy(self, logits,
                   targets,
                   mask = None):
    '''
    logits - tensor, dtype = tf.float32 shape = [..., hidden_size]
    targets - tensor, dtype = tf.float32 shape = [..., hidden_size], or dtype = tf.int32 shape = [...]
    mask - tensor, dtype = tf.float32 shape = [...]
    
    Used to calculate the prediction tensor and accuracy of the logit tensor given the target tensor.
    The mask tensor, where each element is either 0 or 1, can be used to find the correct accuracy if the model if ignoring padding values
    '''
    self.predict = tf.argmax(logits,
                             axis = -1,
                             output_type = tf.int32)
    if targets.dtype != tf.int32:
      targets = tf.argmax(targets,
                          axis = -1,
                          output_type = tf.int32)
    self.correct_prediction = tf.equal(self.predict,
                                       targets)
    if mask is not None:
      self.accuracy = tf.reduce_sum(tf.cast(self.correct_prediction,
                                            dtype = tf.float32) * mask) / tf.reduce_sum(mask)
    else:
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
                                             dtype = tf.float32))
                                             
  def accuracy(self, logits,
               targets,
               mask = None):
    '''
    logits - tensor, dtype = tf.float32 shape = [..., hidden_size]
    targets - tensor, dtype = tf.float32 shape = [..., hidden_size], or dtype = tf.int32 shape = [...]
    mask - tensor, dtype = tf.float32 shape = [...]
    
    Used to calculate the prediction tensor and accuracy of the logit tensor given the target tensor.
    The mask tensor, where each element is either 0 or 1, can be used to find the correct accuracy if the model if ignoring padding values
    '''
    self.predict = tf.argmax(logits,
                             axis = -1,
                             output_type = tf.int32)
    if targets.dtype != tf.int32:
      targets = tf.argmax(targets,
                          axis = -1,
                          output_type = tf.int32)
    self.correct_prediction = tf.equal(self.predict,
                                       targets)
    if mask is not None:
      self.accuracy = tf.reduce_sum(tf.cast(self.correct_prediction,
                                            dtype = tf.float32) * mask) / tf.reduce_sum(mask)
    else:
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
                                             dtype = tf.float32))
      
  def sequential_accuracy(self, logits,
                          targets,
                          mask = None):
    '''
    Similar to accuracy function above, except calculates the sequential accuracy of the logit tensor
    For example, given the logit tensor:
    [[1, 2, 3],
     [4, 5, 7]]
    and the target tensor:
    [[1, 2, 3],
     [4, 5, 6]]
    the accuracy would be calculated as 0.8333, but the sequential accuracy is calculated as 0.5
    '''
    predict = tf.argmax(logits,
                        axis = -1,
                        output_type = tf.int32)
    correct_prediction = tf.equal(predict,
                                  targets)
    if mask is not None:
      correct_prediction = tf.cast(correct_prediction,
                                   dtype = tf.float32)
      correct_prediction *= mask
      correct_prediction += ((mask - 1) * -1)
      correct_prediction = tf.cast(correct_prediction,
                                   dtype = tf.bool)
    self.sequential_accuracy = tf.reduce_all(correct_prediction,
                                             axis = -1)
    self.sequential_accuracy = tf.cast(self.sequential_accuracy,
                                       dtype = tf.float32)
    
def argument():
  class Arg():
    pass
  arg = Arg()
  arg.clip_value_max = 1
  arg.clip_value_min = -1
  arg.norm_value = 1e-3
  arg.stddev_noise = 1e-3
  
  arg.clip_norm = False
  arg.clip_value = False
  arg.grad_noise = False
  return arg

class warmup_learning_rate():
  '''
  A warmup learning rate, introduced in arXiv:1706.03762
  '''
  def __init__(self, lr = None,
               warmup = 0,
               step = 1):
    self.lr = lr
    self.step = step
    self.warmup = warmup
    
  def __call__(self, lr = None,
               step = None,
               warmup = None):
    if lr is not None:
      self.lr = lr
    if step is not None:
      self.step = step
    if warmup is not None:
      self.warmup = warmup
    
    lr = self.lr * min(self.step ** -0.5,
                       self.step * self.warmup ** -1.5)
    
    self.step += 1
    return lr