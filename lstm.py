import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
from utils import *

class LSTM():
  # unidirectional LSTM model
  def __init__(self, dropout_rate,
               embed_size,
               memory_size,
               input_vocab_size,
               layers,
               output_vocab_size = None,
               rnn_cell = 'lstm',
               training = False,
               unidirectional = True,
               n_experts = 15,
               use_mos = False):
    if output_vocab_size is None:
      output_vocab_size = input_vocab_size
               
    self.dropout_rate = dropout_rate
    # dropout rate
    self.embed_size = embed_size
    # the embedding size of the model
    self.memory_size = memory_size
    # the size of each LSTM output
    self.input_vocab_size = input_vocab_size
    # the input vocab size
    self.layers = layers
    # the number of LSTMs performed iteratively
    self.output_vocab_size = output_vocab_size
    # the output vocab size, if set to None then same as input vocab size
    self.training = training
    # whether the model is training or evaluating
    self.use_mos = use_mos
    # whether to use an MoS output or not
    self.rnn_cell = rnn_cell
    # what type of RNN cell to use
    self.n_experts = n_experts
    
    self.unidirectional = unidirectional
    # whether to use unidirectional or bidirectional model
    
    self.model = self.load_model()
    
  def load_model(self):
    inputs = tf.keras.Input(name = 'input',
                            shape = (None, ),
                            batch_size = None,
                            dtype = tf.int32)
    if self.rnn_cell == 'lstm':
      initial_state = tf.keras.Input(name = 'state',
                                     shape = (self.layers, 2, self.memory_size),
                                     batch_size = None,
                                     dtype = tf.float32)
      split_initial_state = tf.split(initial_state,
                                     num_or_size_splits = self.layers,
                                     axis = 1)
      for layer in range(self.layers):
        split_initial_state[layer] = tf.squeeze(split_initial_state[layer],
                                                axis = 1)
        split_initial_state[layer] = tf.split(split_initial_state[layer],
                                              num_or_size_splits = 2,
                                              axis = 1)
        for i in range(2):
          split_initial_state[layer][i] = tf.squeeze(split_initial_state[layer][i],
                                                     axis = 1)
    else:
      initial_state = tf.keras.Input(name = 'state',
                                     shape = (self.layers, self.memory_size),
                                     batch_size = None,
                                     dtype = tf.float32)
      split_initial_state = tf.split(initial_state,
                                     num_or_size_splits = self.layers,
                                     axis = 1)
      for layer in range(self.layers):
        split_initial_state[layer] = tf.squeeze(split_initial_state[layer],
                                                axis = 1)
    embedded_inputs = tf.keras.layers.Embedding(input_dim = self.input_vocab_size,
                                                output_dim = self.embed_size)(inputs)
    embedded_inputs = tf.keras.layers.Dropout(self.dropout_rate)(embedded_inputs,
                                                                 training = self.training)
    rnn_output = embedded_inputs
    new_state = []
    for layer in range(self.layers):
      if layer == 1 and not self.unidirectional:
        rnn_output = tf.reverse(rnn_output,
                                axis = [1])
      if self.rnn_cell == 'lstm':
        rnn_output, state_one, state_two = tf.keras.layers.LSTM(self.memory_size,
                                                                return_sequences = True,
                                                                return_state = True)(rnn_output,
                                                                                     initial_state = split_initial_state[layer])
        new_state.append(tf.concat([tf.expand_dims(state_one,
                                                   axis = 1), tf.expand_dims(state_two,
                                                                             axis = 1)],
                                   axis = 1))
        new_state[-1] = tf.expand_dims(new_state[-1],
                                       axis = 1)
      else:
        rnn_output, state = tf.keras.layers.GRU(self.memory_size,
                                                return_sequences = True,
                                                return_state = True)(rnn_output,
                                                                     initial_state = split_initial_state[layer])
        new_state.append(tf.expand_dims(state,
                                        axis = 1))
      if layer == 1 and not self.unidirectional:
        rnn_output = tf.reverse(rnn_output,
                                axis = [1])
    if self.use_mos:
      predicted_char = MoS(self.output_vocab_size,
                           n_experts = self.n_experts)(rnn_output)
    else:
      predicted_char = tf.keras.layers.Dense(self.output_vocab_size,
                                             activation = tf.nn.softmax)(rnn_output)
    new_state = tf.concat(new_state,
                          axis = 1)
    return tf.keras.Model(inputs = [inputs, initial_state],
                          outputs = [predicted_char, new_state])
                          
  def __call__(self, inputs):
    return self.model(inputs)
    
if __name__ == '__main__':
  inputs = tf.placeholder(tf.int32,
                          [None, None])
  LSTMmodel = LSTM(dropout_rate = 0.1,
                   embed_size = 512,
                   memory_size = 1024,
                   input_vocab_size = 256,
                   layers = 4,
                   rnn_cell = 'gru')
  exit()
  LSTMmodel.model.summary()
  print(inputs)
  outputs = LSTMmodel(inputs)
  print(outputs)