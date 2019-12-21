import numpy as np
import tensorflow as tf

import util_code as utils

def MoS(x,
        vocab_size,
        n_experts = 10):
  '''
  a Mixture-of-Softamx operator, as specified in arXiv:1711.03953
  relatively untested. 
  x - tensor, dtype = tf.float32 shape = [..., hidden_size]
  hidden_size - the final dimension of x
  vocab_size - the vocab_size to calculate
  n_experts - the number of experts, int
  '''
  batch_size = tf.shape(x)[0]
  sequence_size = tf.shape(x)[1]
  hidden_size = tf.shape(x)[2]
  with tf.variable_scope('latent'):
    latent = utils.dense(x,
                         output_dim = n_experts * hidden_size,
                         name = 'latent')
    latent = tf.nn.tanh(latent)
        
  with tf.variable_scope('decoder'):
    latent = tf.reshape(latent,
                        [-1, hidden_size])
    logit = utils.dense(latent,
                        output_dim = vocab_size,
                        name = 'decoder')
        
  with tf.variable_scope('prior'):
    prior_logit = utils.dense(x,
                              output_dim = n_experts,
                              name = 'prior')
    prior_logit = tf.reshape(prior_logit,
                             [-1, n_experts])
    prior = tf.nn.softmax(prior_logit,
                          axis = -1)
    prob = tf.reshape(tf.nn.softmax(tf.reshape(logit,
                                               [-1, vocab_size]),
                                    axis = -1),
                      [-1, n_experts, vocab_size])
    prior = tf.expand_dims(prior,
                           axis = 2)
    prior = tf.tile(prior,
                    [1 ,1, vocab_size])
    prob = (prob * prior)
    prob = tf.reduce_sum(prob,
                         axis = 1)
    prob = tf.reshape(prob,
                      [batch_size, sequence_size, vocab_size])
  return prob

if __name__ == '__main__':
  x = tf.placeholder(tf.float32,
                     [32, 10, 128])
  y = MoS(x,
          10,
          128,
          10000)
  print(y)