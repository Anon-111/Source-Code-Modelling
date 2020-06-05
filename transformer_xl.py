import os
import math
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
from utils import *

class Transformer_XL():
  # XL is capable of seeing across previous sequences by caching them in a memory
  def __init__(self, adaptive_span = False,
               batch_size = 32,
               cache_memory = 'after_final_attention',
               dropout_rate = 0.1,
               embed_size = 256,
               head_size = 64,
               hidden_size = 256,
               input_vocab_size = 256,
               layers = 4,
               max_length = 100,
               max_memory_length = 500,
               memory_length = 256,
               num_heads = 8,
               output_vocab_size = None,
               seq_len = 100,
               tie_weights = True,
               training = True,
               unidirectional = True,
               use_memory = True,
               use_prev = False,
               use_positional_timing = True,
               use_relu = True,
               use_signal_timing = False,
               **args):
    self.adaptive_span = adaptive_span
    self.batch_size = batch_size
    self.cache_memory = cache_memory
    # the point of the block to cache the memory
    self.dropout_rate = dropout_rate
    self.embed_size = embed_size
    self.head_size = head_size
    self.hidden_size = hidden_size
    self.input_vocab_size = input_vocab_size
    self.layers = layers
    self.max_length = max_length
    self.max_memory_length = max_memory_length
    # if the cached memory supersedes this length, then it is sliced
    # this limits the computational cost of the model
    self.memory_length = memory_length
    # set to None for dynamic
    self.num_heads = num_heads
    self.output_vocab_size = output_vocab_size
    self.seq_len = seq_len
    self.tie_weights = tie_weights
    # whether to tie the weights across all layers for the relative bias in the attention mechanism
    self.training = training
    self.unidirectional = unidirectional
    self.use_memory = use_memory
    # whether to use an XL-memory
    self.use_positional_timing = use_positional_timing
    self.use_prev = use_prev
    # whether the XL-memory is the previous ffd_output, or whether to concatenate the ffd_output onto the old memory
    self.use_relu = use_relu
    self.use_signal_timing = use_signal_timing
    
    self.model = self.build_model()
    
  def __call__(self, inputs, 
               memory = None):
    if self.use_memory:
      return self.model([inputs, memory])
    else:
      return self.model(inputs)
  
  def cache(self, old_memory,
            new_memory):
    if not self.use_prev:
      new_memory = tf.concat([old_memory, new_memory],
                             axis = 1)
      if self.max_memory_length:
        new_memory = new_memory[:,-1 * self.max_memory_length:]
    return tf.stop_gradient(tf.expand_dims(new_memory,
                                           axis = 1))

  def save_model(self, name):
    self.model.save(name)
    
  def load_model(self, name):
    self.model.load_weights(name)
  
  def build_model(self):
    activation = tf.nn.relu if self.use_relu else gelu
  
    if self.output_vocab_size is None:
      self.output_vocab_size = self.input_vocab_size
    
    if self.tie_weights:
      # the relative attention requires trainable relative bias vectors
      # these biases can either be used per layer, or can be a set of global vectors used for each attention mechanism
      r_w_bias, r_r_bias = Relative_Bias(num_heads = self.num_heads,
                                         head_size = self.head_size)(None)
    else:
      r_w_bias, r_r_bias = None, None
     
    inputs = tf.keras.Input(shape = (self.seq_len, ),
                            batch_size = self.batch_size,
                            dtype = tf.int32)
    if self.use_memory:
      memory = tf.keras.Input(shape = (self.layers, self.memory_length, self.hidden_size), 
                              batch_size = self.batch_size,
                              dtype = tf.float32)
      # list for the memory to be cached and concatenated at the end of the model's build
      new_memory = []
    else:
      memory = None
      
    embedded_inputs = tf.keras.layers.Embedding(input_dim = self.input_vocab_size,
                                                output_dim = self.embed_size)(inputs)
    if self.embed_size != self.hidden_size:
      embedded_inputs = tf.keras.layers.Dense(self.hidden_size,
                                              activation = activation)(embedded_inputs)
    dropout_embedded_inputs = tf.keras.layers.Dropout(self.dropout_rate)(embedded_inputs,
                                                                         training = self.training)
    if self.use_positional_timing:
      dropout_embedded_inputs = Positional_Embedding(max_length = self.max_length)(dropout_embedded_inputs)
    if self.use_signal_timing:
      dropout_embedded_inputs = Timing_Signal()(dropout_embedded_inputs)
    ffd_output = dropout_embedded_inputs
    for layer in range(self.layers):
      # for Tr-I, the block starts with a layer norm
      # to maintain the correct normalization for both the inputs and the memory
      # the layer normalization is conducted inside the Relative_Multihead_Attention layer
      if self.use_memory:
        # in arXiv:1901.02860, then the memory is cached immediately before the attention mechanism
        if self.cache_memory == 'before_attention':
          new_memory.append(self.cache(memory[:,layer],
                                       ffd_output))
        elif self.cache_memory == 'after_attention' and layer != 0:
          new_memory.append(self.cache(memory[:,layer - 1],
                                       ffd_output))
      attention_output = Relative_Multihead_Attention(tie_weights = self.tie_weights,
                                                      head_size = self.head_size,
                                                      unidirectional = self.unidirectional,
                                                      adaptive_span = self.adaptive_span,
                                                      num_heads = self.num_heads)(ffd_output,
                                                                                  training = self.training,
                                                                                  dropout_rate = self.dropout_rate,
                                                                                  r_w_bias = r_w_bias,
                                                                                  r_r_bias = r_r_bias,
                                                                                  memory = memory[:,layer] if self.use_memory else None)
      attention_output = tf.keras.layers.Dropout(self.dropout_rate)(attention_output,
                                                                    training = self.training)
      attention_output += ffd_output
      ffd_output = tf.keras.layers.LayerNormalization()(attention_output)
      ffd_output = Transformer_FFD(activation = activation)(ffd_output,
                                                            training = self.training,
                                                            dropout_rate = self.dropout_rate)
      ffd_output = tf.keras.layers.Dropout(self.dropout_rate)(ffd_output,
                                                              training = self.training)
      ffd_output += attention_output
      # FLAGS: cache is here
      ffd_output = tf.keras.layers.LayerNormalization()(ffd_output)
    transformer_output = tf.keras.layers.Dense(self.output_vocab_size,
                                               activation = 'softmax')(ffd_output)
    if self.use_memory:
      if self.cache_memory == 'after_attention':
        new_memory.append(self.cache(memory[:,-1],
                                     ffd_output))
      elif self.cache_memory == 'after_final_attention':
        for l in range(self.layers):
          new_memory.append(self.cache(memory[:,l],
                                       ffd_output))
      new_memory = tf.concat(new_memory,
                             axis = 1)
      return tf.keras.Model(inputs = [inputs, memory],
                            outputs = [transformer_output, new_memory])
    else:
      return tf.keras.Model(inputs = [inputs],
                            outputs = [transformer_output])

if __name__ == '__main__':
  model = Transformer_XL()
  inputs = tf.placeholder(tf.int32, [32, 100])
  memory = tf.placeholder(tf.float32, [32, 4, 256, 256])
  print(model(inputs, memory))
  exit()
  trainable_variables = sum([np.prod(variable.shape) for variable in model.trainable_variables])
  print(trainable_variables)
  exit()
  
  inputs = tf.placeholder(tf.int32, [32, 100])
  memory = tf.placeholder(tf.float32, [32, 6, 256, 256])
  output = model(inputs, memory)
  print(output)
  exit()
  
  variables = model.model.variables
  for variable in variables:
    if 'relative__bias' in variable.name:
      print(variable)
      print(tf.gradients(output, variable))