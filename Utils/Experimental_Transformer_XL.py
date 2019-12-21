import numpy as np
import tensorflow as tf
import functools

import sys
sys.path.append('/home/tom/Desktop/Programming/Models/Utils')
import util_code as utils
import loss
import optimize
import mos
import develop_bias

class Experimental_Transformer_XL():
  def __init__(self, arg,
               name = None):
    if name:
      self.name = name
    else:
      if arg.use_recurrent_encoder:
        print('Extended-Reurrent Transformer')
      self.name = 'Experimental-Transformer-XL'
    batch_size = 128
    input_sequence_size = 10
    memory_sequence_size = 15
    if __name__ != '__main__':
      batch_size = input_sequence_size = memory_sequence_size = None
    self.arg = arg
    self.inputs = tf.placeholder(tf.int32,
                                 shape = [batch_size, input_sequence_size], # (batch_size, input_sequence_size)
                                 name = 'inputs')
    if self.arg.classification:
      self.targets = tf.placeholder(tf.int32,
                                    shape = [batch_size], # (batch_size, output_sequence_size)
                                    name = 'targets')
    else:
      self.targets = tf.placeholder(tf.int32,
                                    shape = [batch_size, input_sequence_size], # (batch_size, input_sequence_size)
                                    name = 'targets')
    self.memory = tf.placeholder(tf.float32,
                                 shape = [self.arg.encoder_layers, batch_size, memory_sequence_size, self.arg.hidden_size],
                                 name = 'memory')
    self.training = tf.placeholder(tf.bool)
    self.keep_prob = tf.placeholder(tf.float32)
    self.learning_rate = tf.placeholder(tf.float32)
    self.batch_size = tf.shape(self.inputs)[0]
    self.input_sequence_size = tf.shape(self.inputs)[1]
    self.memory_sequence_size = tf.shape(self.memory)[2]
    
    self.encoder_self_attention_bias = develop_bias._create_mask(self.input_sequence_size,
                                                                 self.arg.unidirectional_encoder)
    if self.arg.use_XL_attention:
      self.encoder_self_attention_bias = tf.concat([tf.zeros([1, 1, self.input_sequence_size, self.memory_sequence_size]), self.encoder_self_attention_bias],
                                                   axis = -1)
    
    if self.arg.mask_loss:
      if self.arg.classification:
        self.loss_mask = tf.placeholder(tf.float32,
                                        shape = [batch_size],
                                        name = 'loss_mask')
      else:
        self.loss_mask = tf.placeholder(tf.float32,
                                        shape = [batch_size, input_sequence_size], # (batch_size, input_sequence_size)
                                        name = 'loss_mask')
    else:
      self.loss_mask = None
      
    self.new_mems = []
    
    if self.arg.ffd == 'transformer_ffd':
      self.ffd = self.transformer_ffd
    elif self.arg.ffd == 'sru':
      from SRU import SRU
      self.ffd = SRU
    elif self.arg.ffd == 'sepconv':
      self.ffd = self.sepconv
      
    if 'stop' in self.arg.pos:
      embedding_size = self.arg.hidden_size - 1
    else:
      embedding_size = self.arg.hidden_size
    with tf.variable_scope('encoder_embedding'):
      encoder_input, enc_params = utils.embedding(self.inputs,
                                                  model_dim = embedding_size,
                                                  vocab_size = self.arg.input_vocab_size,
                                                  name = 'encode')
      params = enc_params
    
    with tf.variable_scope('positional_encoding'):
      with tf.variable_scope('encoder'):
        encoder_input = self.timing_position(encoder_input)
      
    with tf.variable_scope('encoder'):
      encoder_input = self.dropout_fn(encoder_input)
      encoder_output = self.encoder(encoder_input,
                                    encoder_self_attention_bias = self.encoder_self_attention_bias)
      self.new_mems = tf.stack(self.new_mems)
    
    if self.arg.classification:
      encoder_output = encoder_output[:,-1]
    with tf.variable_scope('output'):
      #weights = tf.transpose(params[0],
      #                       [1, 0])
      weights = tf.get_variable('weights',
                                shape = [self.arg.hidden_size, self.arg.target_vocab_size],
                                dtype = tf.float32)
      bias = tf.get_variable('bias',
                             shape = [self.arg.target_vocab_size],
                             dtype = tf.float32)
      self.logits = tf.tensordot(encoder_output,
                                 weights,
                                 axes = 1) + bias
      self.loss_cl = loss.Loss(self.logits,
                               self.targets,
                               self.arg.loss,
                               vocab_size = self.arg.target_vocab_size,
                               label_smoothing = self.arg.label_smoothing)
      cost = self.loss_cl.loss
    if self.arg.mask_loss:
      self.cost = tf.reduce_mean(cost * self.loss_mask)
    else:
      self.cost = tf.reduce_mean(cost)
    if self.arg.weight_decay_regularization:
      l2_loss = self.loss_cl.l2_loss(tf.trainable_variables())
      l2_loss *= self.arg.weight_decay_hyperparameter
      self.cost += l2_loss
    self.optimizer = optimize.Optimizer(arg,
                                        loss = self.cost,
                                        learning_rate = self.learning_rate)
    self.optimizer.accuracy(self.logits,
                            self.targets,
                            mask = self.loss_mask)
    self.train_op = self.optimizer.train_op
    self.predict = self.optimizer.predict
    self.correct_prediction = self.optimizer.correct_prediction
    self.accuracy = self.optimizer.accuracy
    self.optimizer.sequential_accuracy(self.logits,
                                       self.targets,
                                       mask = self.loss_mask)
    self.sequential_accuracy = self.optimizer.sequential_accuracy
    self.fetches = [encoder_input, encoder_output, self.logits]
      
  def encoder(self, inputs,
              encoder_self_attention_bias):
    self.prepare_attention()
    
    x = inputs
    recurr_x = inputs
    for layer in range(1,
                       self.arg.encoder_layers + 1):
      with tf.variable_scope('layer_{}'.format(layer)):
        x = self.timing_position(x)
        y = self.XL_attention(x,
                              layer)
        if self.arg.use_active_memory:
          z = self.highway_convolution(x)
          x = y + z
        else:
          x = y
        
        with tf.variable_scope('ffd'):
          y = utils.layer_norm(x)
          y = self.ffd(y)
          y = self.dropout_fn(y)
          x += y
          
    if self.arg.use_recurrent_encoder:
      recurr_x = self.recurrent_encoder(recurr_x)
      x, recurr_x = utils.layer_norm(x), utils.layer_norm(recurr_x)
      x += recurr_x
    else:
      x = utils.layer_norm(x)
    return x
    
  def transformer_ffd(self, x):
    x = utils.dense(x,
                    output_dim = self.arg.filter_size,
                    use_bias = True,
                    name = 'ffd_1')
    x = self.dropout_fn(x)
    if self.arg.use_relu:
      x = tf.nn.relu(x)
    else:
      x = utils.gelu(x)
    return utils.dense(x,
                       output_dim = self.arg.hidden_size,
                       use_bias = True,
                       name = 'ffd_2')
  
  def prepare_attention(self):
    with tf.variable_scope('positional_embedding'):
      pos_seq = tf.range(tf.shape(self.encoder_self_attention_bias)[-1] - 1,
                         -1,
                         -1.0)
      inv_freq = 1 / (10000 ** (tf.range(0, self.arg.hidden_size, 2.0) / self.arg.hidden_size))
      sinusoid_inp = tf.einsum('i,j->ij',
                               pos_seq,
                               inv_freq)
      pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)],
                          axis = -1)
      self.pos_emb = tf.tile(pos_emb[None,:,:],
                             [self.batch_size, 1, 1])
      if self.arg.tie_weights:
        self.r_w_bias = tf.get_variable('r_w_bias',
                                        shape = [1, self.arg.num_heads, 1, self.arg.head_size],
                                        dtype = tf.float32)
        self.r_r_bias = tf.get_variable('r_r_bias',
                                        shape = [1, self.arg.num_heads, 1, self.arg.head_size],
                                        dtype = tf.float32)
      else:
        self.r_w_bias = tf.get_variable('r_w_bias',
                                        shape = [self.arg.encoder_layers, 1, self.arg.num_heads, 1, self.arg.head_size],
                                        dtype = tf.float32)
        self.r_r_bias = tf.get_variable('r_r_bias',
                                        shape = [self.arg.encoder_layers, 1, self.arg.num_heads, 1, self.arg.head_size],
                                        dtype = tf.float32)
        
  def XL_attention(self, x,
                   layer):
    with tf.variable_scope('attention'):
      if self.arg.save_mems_at_start:
        self.new_mems.append(self._cache_mem(x, 
                                             self.memory[layer - 1]))
      if self.arg.per_layer_memory:
        memory = tf.concat([self.memory[layer - 1], x],
                           axis = 1)
      elif not self.arg.per_layer_memory:
        memory = tf.concat([self.memory[-1], x], axis = 1)
      y = utils.layer_norm(x)
      memory = utils.layer_norm(memory)
      if not self.arg.use_XL_attention:
        memory = None
      q, k, v = utils.compute_qkv(query = y,
                                  memory = memory,
                                  total_key_depth = self.arg.head_size * self.arg.num_heads,
                                  total_value_depth = self.arg.head_size * self.arg.num_heads,
                                  deparameterize = self.arg.deparameterize)
      r = utils.dense(self.pos_emb,
                      output_dim = self.arg.head_size * self.arg.num_heads,
                      use_bias = False,
                      name = 'pos_emb')
      r = tf.reshape(r,
                     [self.batch_size, self.arg.num_heads, -1, self.arg.head_size])
      q = utils.split_heads(q,
                            self.arg.num_heads)
      k = utils.split_heads(k,
                            self.arg.num_heads)
      v = utils.split_heads(v,
                            self.arg.num_heads)
      if self.arg.tie_weights:
        AD = tf.matmul(q + self.r_w_bias,
                       k,
                       transpose_b = True)
        BD = tf.matmul(q + self.r_r_bias,
                       r,
                       transpose_b = True)
      else:
        AD = tf.matmul(q + self.r_w_bias[layer - 1],
                       k,
                       transpose_b = True)
        BD = tf.matmul(q + self.r_r_bias[layer - 1],
                       r,
                       transpose_b = True)
        
      BD = self.rel_shift(BD) 
      logits = AD + BD
      logits /= k.shape.as_list()[-1]
      logits += self.encoder_self_attention_bias
      weights = tf.nn.softmax(logits,
                              name = 'attention_weights')
      y = tf.matmul(weights,
                    v)
      y = utils.combine_heads(y)
      y.set_shape(y.shape.as_list()[:-1] + [self.arg.head_size * self.arg.num_heads])
      with tf.variable_scope('output'):
        y = utils.dense(y,
                        output_dim = self.arg.hidden_size,
                        use_bias = False,
                        name = 'output_transform')
      y = self.dropout_fn(y)
      x += y
      if self.arg.save_mems_at_start == False:
        self.new_mems.append(self._cache_mem(x, 
                                             self.memory[layer - 1]))
      return x
    
  def recurrent_encoder(self, recurr_x):
    for layer in range(1,
                       self.arg.rnn_encoder_layers + 1):
      with tf.variable_scope('RNN_layer_{}'.format(layer)):
        recurr_x = utils.layer_norm(recurr_x)
        cell = tf.nn.rnn_cell.GRUCell(self.arg.hidden_size,
                                      name = 'fwd_cell')
        if not self.arg.unidirectional_encoder:
          cell_fwd = cell
          cell_bwd = tf.nn.rnn_cell.GRUCell(self.arg.hidden_size,
                                            name = 'bwd_cell')
          outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fwd,
                                                           cell_bwd,
                                                           recurr_x,
                                                           dtype = tf.float32)
          recurr_x = outputs[0] + outputs[1]
        else:
          recurr_x, _ = tf.nn.dynamic_rnn(cell,
                                          recurr_x,
                                          dtype = tf.float32)
        recurr_y = utils.layer_norm(recurr_x)
        recurr_y = self.ffd(recurr_y)
        recurr_x += recurr_y
    return recurr_x
  
  def convolution(self, query,
                  act_fn,
                  input_size = None,
                  hidden_size = None):
    if input_size == None:
      input_size = query.shape.as_list()[-1]
    if hidden_size == None:
      hidden_size = self.arg.hidden_size
    with tf.variable_scope('convolution'):
      weights = tf.get_variable('weights',
                                shape = [self.arg.kernel, input_size, hidden_size],
                                dtype = tf.float32)
      bias = tf.get_variable('bias',
                             shape = [hidden_size],
                             dtype = tf.float32)
      if self.arg.unidirectional_encoder:
        query = tf.concat([tf.zeros([self.batch_size, self.arg.kernel - 1, input_size]), query],
                        axis = 1)
        return act_fn(tf.nn.convolution(query,
                                        weights,
                                        padding = 'VALID') + bias)
      else:
        return act_fn(tf.nn.convolution(query,
                                        weights,
                                        padding = 'SAME') + bias)
    
  def highway_convolution(self, query):
    with tf.variable_scope('convolution_one'):
      a = self.convolution(query,
                           tf.identity)
    with tf.variable_scope('convolution_two'):
      b = self.convolution(query,
                           self.sigmoid_cutoff)
    return tf.multiply(a, b) + tf.multiply(query, 1 - b)
  
  def sigmoid_cutoff(self, state):
    return tf.maximum(tf.minimum(1.2 * tf.sigmoid(state) - 0.1,
                                 1.0),
                      0.0)
        
  def dropout_fn(self, x,
                 keep_prob = None):
    return tf.cond(self.training,
                   lambda: utils.dropout(x,
                                         keep_prob = self.keep_prob,
                                         dropout = self.arg.dropout_type),
                   lambda: tf.identity(x))
  
  def sepconv(self, x):
    output = utils.separable_convolution_2d(x,
                                            hidden_size = self.arg.filter_size,
                                            kernel_size = 3,
                                            name = 'conv1')
    if self.arg.use_relu:
      output = tf.nn.relu(output)
    else:
      output = utils.gelu(output)
    output = self.dropout_fn(output)
    return utils.separable_convolution_2d(output,
                                          hidden_size = self.arg.hidden_size,
                                          kernel_size = 5,
                                          name = 'conv2')
  
  def timing_position(self, inputs):
    sequence_size = tf.shape(inputs)[1]
    
    if self.arg.pos == 'timing':
      return inputs + utils.add_timing_signal_1d(sequence_size = sequence_size,
                                                 channels = self.arg.hidden_size)
    elif self.arg.pos == 'emb':
      return inputs + utils.add_positional_embedding(inputs,
                                                     max_length = self.arg.input_max_length, ###
                                                     hidden_size = self.arg.hidden_size,
                                                     input_sequence_size = sequence_size,
                                                     name = 'positional_embedding')
    elif self.arg.pos == 'linear_stop':
      sequence_size = tf.shape(inputs)[1]
      batch_size = tf.shape(inputs)[0]
      stop = tf.range(sequence_size)
      
      stop /= sequence_size
      
      stop = tf.expand_dims(stop,
                            axis = 0)
      stop = tf.tile(stop,
                     [batch_size, 1])
      stop = tf.cast(tf.expand_dims(stop,
                                    axis = 2),
                     dtype = tf.float32)
      return tf.concat([inputs, stop],
                       axis = -1)
    elif self.arg.pos == 'tanh_stop':
      sequence_size = tf.shape(inputs)[1]
      batch_size = tf.shape(inputs)[0]
      stop = tf.range(sequence_size)
      stop = tf.cast(stop,
                     dtype = tf.float32)
      sequence_size = tf.cast(sequence_size,
                              dtype = tf.float32)
      
      gamma = 3.0
      stop = tf.nn.tanh(gamma * stop/sequence_size) + 1 - tf.nn.tanh(gamma)
      
      stop = tf.expand_dims(stop,
                            axis = 0)
      stop = tf.tile(stop,
                     [batch_size, 1])
      stop = tf.expand_dims(stop,
                            axis = 2)
      return tf.concat([inputs, stop],
                       axis = -1)
    elif self.arg.pos == 'exp_stop':
      sequence_size = tf.shape(inputs)[1]
      batch_size = tf.shape(inputs)[0]
      stop = tf.range(sequence_size)
      stop = tf.cast(stop,
                     dtype = tf.float32)
      sequence_size = tf.cast(sequence_size,
                              dtype = tf.float32)
      
      gamma = 3.0
      stop = tf.exp(gamma * (stop - sequence_size) / sequence_size)
      
      stop = tf.expand_dims(stop,
                            axis = 0)
      stop = tf.tile(stop,
                     [batch_size, 1])
      stop = tf.expand_dims(stop,
                            axis = 2)
      return tf.concat([inputs, stop],
                       axis = -1)
    else:
      return inputs
    
  def rel_shift(self, x):
    x_shape = tf.shape(x)
    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
    x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2]])
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_shape)
    return x
  
  def _cache_mem(self, curr_out, 
                 prev_mem):
    if self.arg.use_prev:
      return tf.stop_gradient(curr_out)
    new_mem = tf.concat([prev_mem, curr_out], 
                        axis = 1)
    if self.arg.mem_len:
      new_mem = new_mem[:,-self.arg.mem_len:]
    return tf.stop_gradient(new_mem)
  
def argument():
  arg = optimize.argument()
  arg.dropout_type = 'vanilla' # 'vanilla', 'broadcast', 'alpha'
  arg.ffd = 'transformer_ffd' # 'transformer_ffd' 'sru' 'sepconv'
  arg.loss = 'sparse_softmax_cross_entropy_with_logits'
  arg.pos = 'emb' # 'timing' 'emb' 'linear_stop' 'tanh_stop' 'exp_stop'
  
  arg.encoder_layers = 2
  arg.filter_size = 1024
  arg.head_size = 64
  arg.hidden_size = 256
  arg.input_max_length = 10
  arg.input_vocab_size = 1000
  arg.kernel = 32
  arg.label_smoothing = 1.0
  arg.max_relative_position = 100
  arg.mem_len = 200
  arg.num_heads = 8
  arg.rnn_encoder_layers = 2
  arg.target_vocab_size = 1000
  arg.weight_decay_hyperparameter = 0.001
  
  arg.classification = False
  arg.deparameterize = False
  arg.mask_loss = True
  arg.per_layer_memory = True
  arg.save_mems_at_start = True
  arg.tie_weights = True
  arg.unidirectional_encoder = True
  arg.use_active_memory = False
  arg.use_mos = False
  arg.use_prev = False
  arg.use_recurrent_encoder = True
  arg.use_relu = True
  arg.use_XL_attention = True
  arg.weight_decay_regularization = False
  return arg
  
if __name__ == '__main__':
  arg = argument()
  
  model = Experimental_Transformer_XL(arg)