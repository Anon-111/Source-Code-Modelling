import numpy as np
import tensorflow as tf
import functools

import util_code as utils
import loss
import optimize
import develop_bias

class Transformer_XL():
  def __init__(self, arg,
               name = None):
    '''
    Transformer-XL, introduced in arXiv:1901.02860
    code was based off https://github.com/kimiyoung/transformer-xl/blob/master/tf/model.py, but adjusted for simplicity
    
    # DYNAMIC EVALUATION - https://github.com/benkrause/dynamiceval-transformer
    
    '''
    if name:
      self.name = name
    else:
      self.name = 'Transformer-XL'
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
    if self.arg.calculate_loss:
      with tf.variable_scope('output'):
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
    else:
      self.encoder_output = encoder_output
      
  def encoder(self, inputs,
              encoder_self_attention_bias):
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
      pos_emb = tf.tile(pos_emb[None,:,:],
                        [self.batch_size, 1, 1])
      if self.arg.tie_weights:
        r_w_bias = tf.get_variable('r_w_bias',
                                   shape = [1, self.arg.num_heads, 1, self.arg.head_size],
                                   dtype = tf.float32)
        r_r_bias = tf.get_variable('r_r_bias',
                                   shape = [1, self.arg.num_heads, 1, self.arg.head_size],
                                   dtype = tf.float32)
      else:
        r_w_bias = tf.get_variable('r_w_bias',
                                   shape = [self.arg.encoder_layers, 1, self.arg.num_heads, 1, self.arg.head_size],
                                   dtype = tf.float32)
        r_r_bias = tf.get_variable('r_r_bias',
                                   shape = [self.arg.encoder_layers, 1, self.arg.num_heads, 1, self.arg.head_size],
                                   dtype = tf.float32)
    
    x = inputs
    for layer in range(1,
                       self.arg.encoder_layers + 1):
      with tf.variable_scope('layer_{}'.format(layer)):
        x = self.timing_position(x)
        with tf.variable_scope('attention'):
          
          self.new_mems.append(self._cache_mem(x, 
                                               self.memory[layer - 1]))
          
          memory = tf.concat([self.memory[layer - 1], x],
                             axis = 1)
          y = utils.layer_norm(x)
          memory = utils.layer_norm(memory)
          q, k, v = utils.compute_qkv(query = y,
                                      memory = memory,
                                      total_key_depth = self.arg.head_size * self.arg.num_heads,
                                      total_value_depth = self.arg.head_size * self.arg.num_heads,
                                      deparameterize = self.arg.deparameterize)
          r = utils.dense(pos_emb,
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
            AD = tf.matmul(q + r_w_bias,
                           k,
                           transpose_b = True)
            BD = tf.matmul(q + r_r_bias,
                           r,
                           transpose_b = True)
          else:
            AD = tf.matmul(q + r_w_bias[layer - 1],
                           k,
                           transpose_b = True)
            BD = tf.matmul(q + r_r_bias[layer - 1],
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
          
        
        with tf.variable_scope('ffd'):
          y = utils.layer_norm(x)
          y = self.ffd(y)
          y = self.dropout_fn(y)
          x += y
    with tf.variable_scope('output'):
      return utils.layer_norm(x)
    
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
  arg.dropout_type = 'vanilla' # 'vanilla', 'alpha'
  arg.ffd = 'transformer_ffd' # 'transformer_ffd' 'sru' 'sepconv'
  arg.loss = 'sparse_softmax_cross_entropy_with_logits'
  arg.pos = 'emb' # 'timing' 'emb' 'linear_stop' 'tanh_stop' 'exp_stop'
  
  arg.encoder_layers = 2 # the number of encoder layers
  arg.filter_size = 1024 # the filter size
  arg.head_size = 64 # the size of each head in the attention mechanisms
  arg.hidden_size = 256 # the hidden size
  arg.input_max_length = 10 # the maximum sequence size of the input, for the 'emb' pos
  arg.input_vocab_size = 1000 # the vocab size for the input
  arg.label_smoothing = 1.0 # the hyperparameter for label smoothing
  arg.mem_len = 200 # the length of memory tensor
  arg.num_heads = 8 # the number of heads for the attention mechanisms
  arg.target_vocab_size = 1000 # the vocab size for the targets
  arg.weight_decay_hyperparameter = 0.001 # the hyperparameter for weight decay
  
  arg.calculate_loss = True
  arg.classification = True # whether the final output is a sequence, or single label
  arg.deparameterize = False # KEEP AS FALSE
  arg.mask_loss = True # whether parts of the loss is masked
  arg.unidirectional_encoder = False # whether the encoder is unidirectional
  arg.use_prev = False # whether the previous memory is used entirely, only partially
  arg.use_relu = True # whether the activation functions are ReLU or GELU
  arg.tie_weights = True # whether the attention weights are tied across time
  arg.weight_decay_regularization = False # whether to use weight decay
  return arg

if __name__ == '__main__':
  arg = argument()
  
  model = Transformer_XL(arg)