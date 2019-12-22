import numpy as np
import tensorflow as tf
import functools

import util_code as utils
import loss
import optimize
import develop_bias

class Evolved_Transformer():
  def __init__(self, arg,
               name = None):
    '''
    The Evolved Transformer, introduced arXiv:1901.11117, has an architecture designed by an evolutionary algorithm. 
    Across many experiments, the Evolved Transformer appears to outperform traditional Transformer
    '''
    batch_size = 32
    input_sequence_size = 10
    output_sequence_size = 12
    if __name__ != '__main__':
      batch_size = input_sequence_size = output_sequence_size = None
    if name:
      self.name = name
    else:
      self.name = 'Evolved_Transformer'
    self.arg = arg
    self.inputs = tf.placeholder(tf.int32,
                                 shape = [batch_size, input_sequence_size], # (batch_size, input_sequence_size)
                                 name = 'inputs')
    if self.arg.classification:
      self.targets = tf.placeholder(tf.int32,
                                    shape = [batch_size],
                                    name = 'targets')
    else:
      self.targets = tf.placeholder(tf.int32,
                                    shape = [batch_size, output_sequence_size], # (batch_size, output_sequence_size)
                                    name = 'targets')
    self.training = tf.placeholder(tf.bool)
    self.keep_prob = tf.placeholder(tf.float32)
    self.learning_rate = tf.placeholder(tf.float32)
    self.batch_size = tf.shape(self.inputs)[0]
    self.input_sequence_size = tf.shape(self.inputs)[1]
    if not self.arg.classification:
      self.target_sequence_size = tf.shape(self.targets)[1]
    if self.arg.mask_loss:
      if self.arg.classification:
        self.loss_mask = tf.placeholder(tf.float32,
                                        shape = [batch_size],
                                        name = 'loss_mask')
      else:
        self.loss_mask = tf.placeholder(tf.float32,
                                        shape = [batch_size, output_sequence_size], # (batch_size, output_sequence_size)
                                        name = 'loss_mask')
    else:
      self.loss_mask = None
    self.encoder_self_attention_bias = develop_bias._create_mask(self.input_sequence_size,
                                                                 self.arg.unidirectional_encoder)
    if not self.arg.classification:
      self.encoder_decoder_attention_bias = tf.zeros([1, 1, self.target_sequence_size, self.input_sequence_size],
                                                     name = 'encoder_self_attention_bias')
      self.decoder_self_attention_bias = develop_bias._create_mask(self.target_sequence_size,
                                                                   self.arg.unidirectional_decoder)
      
    if 'stop' in self.arg.pos:
      embedding_size = self.arg.hidden_size - 1
    else:
      embedding_size = self.arg.hidden_size
    with tf.variable_scope('encoder_embedding'):
      encoder_input, enc_params = utils.embedding(self.inputs,
                                                  model_dim = embedding_size,
                                                  vocab_size = self.arg.input_vocab_size,
                                                  name = 'encode')
    with tf.variable_scope('decoder_embedding'):
      decoder_input, dec_params = utils.embedding(self.targets,
                                                  model_dim = embedding_size,
                                                  vocab_size = self.arg.target_vocab_size,
                                                  name = 'decode')
    if self.arg.use_decoder:
      params = dec_params
      del enc_params
    else:
      params = enc_params
      del dec_params
    
    with tf.variable_scope('positional_encoding'):
      with tf.variable_scope('encoder'):
        encoder_input = self.timing_position(encoder_input)
      with tf.variable_scope('decoder'):
        if not self.arg.classification:
          decoder_input = self.timing_position(decoder_input)
    
    with tf.variable_scope('encoder'):
      encoder_input = self.dropout_fn(encoder_input)
      encoder_output = self.encoder(encoder_input,
                                    encoder_self_attention_bias = self.encoder_self_attention_bias)
    if self.arg.use_decoder:
      with tf.variable_scope('decoder'):
        decoder_input = tf.pad(decoder_input,
                               paddings = [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        decoder_input = self.dropout_fn(decoder_input)
        decoder_output = self.decoder(decoder_input,
                                      encoder_output,
                                      decoder_self_attention_bias = self.decoder_self_attention_bias,
                                      encoder_decoder_attention_bias = self.encoder_decoder_attention_bias)
    if self.arg.use_decoder:
      if self.arg.classification:
        output = decoder_output[:,-1]
      else:
        output = decoder_output
    else:
      if self.arg.classification:
        output = encoder_output[:,-1]
      else:
        output = encoder_output
    with tf.variable_scope('output'):
      weights = tf.get_variable('weights',
                                shape = [self.arg.hidden_size, self.arg.target_vocab_size],
                                dtype = tf.float32)
      bias = tf.get_variable('bias',
                             shape = [self.arg.target_vocab_size],
                             dtype = tf.float32)
      if arg.use_decoder:
        self.logits = tf.tensordot(output,
                                   weights,
                                   axes = 1) + bias
      else:
        self.logits = tf.tensordot(output,
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
    if self.arg.adaptive_mask:
      self.encoder_l0 = tf.reduce_sum(self.encoder_l0)
      self.cost += 0.0001 * self.encoder_l0
      if self.arg.use_decoder:
        self.decoder_l0 = tf.reduce_sum(self.decoder_l0)
        self.cost += 0.0001 * self.decoder_l0
    if self.arg.weight_decay_regularization:
      l2_loss = self.loss_cl.l2_loss(tf.trainable_variables())
      l2_loss *= self.arg.weight_decay_hyperparameter
      self.cost += l2_loss
    self.optimizer = optimize.Optimizer(arg,
                                        loss = self.cost,
                                        learning_rate = self.learning_rate)
    self.train_op = self.optimizer.train_op
    self.optimizer.accuracy(self.logits,
                            self.targets,
                            mask = self.loss_mask)
    self.predict = self.optimizer.predict
    self.correct_prediction = self.optimizer.correct_prediction
    self.accuracy = self.optimizer.accuracy
    self.optimizer.sequential_accuracy(self.logits,
                                       self.targets,
                                       mask = self.loss_mask)
    self.sequential_accuracy = self.optimizer.sequential_accuracy
      
  def encoder(self, inputs,
              encoder_self_attention_bias):
    x = inputs
    if self.arg.adaptive_mask:
      self.encoder_l0 = []
    for layer in range(1,
                       self.arg.encoder_layers + 1):
      with tf.variable_scope('layer_{}'.format(layer)):
        with tf.variable_scope('gated_linear_unit'):
          y = utils.layer_norm(x)
          y = utils.convolution_gating(y,
                                       kernel_size = 1,
                                       input_dim = y.shape.as_list()[-1],
                                       output_dim = y.shape.as_list()[-1])
          y = self.dropout_fn(y)
          x += y
        with tf.variable_scope('conv_branches'):
          y = utils.layer_norm(x)
          if self.arg.use_relu:
            left_state = tf.nn.relu(utils.dense(y,
                                                output_dim = int(self.arg.hidden_size * 4),
                                                name = 'left_branch'))
          else:
            left_state = utils.gelu(utils.dense(y,
                                                output_dim = int(self.arg.hidden_size * 4),
                                                name = 'right_branch'))
          left_state = self.dropout_fn(left_state)
          with tf.variable_scope('right_branch'):
            '''
            given that the tensor, at this point, is of shape [batch_size, sequence_size, hidden_size], 
            and the kernel size of the convolution is 3, 
            then an unmoderated convolution, at time-step t, would analyze the time-steps (t-1, t, t+1) for the output of time-step t
            If the anaylsis is unidirectional, and that the analysis at time-step t cannot see 'ahead through time', this form of analysis if invalid.
            Therefore, in order to avoid this illegal analysis, a zeros vector is concatenated to the left of the tensor.
            Therefore, at time-step t, the time-steps (t-2, t-1, t) is analyzed, where the tokens at -2 and -1 are 0
            If the analysis is bidirectional, analyzing the time-steps (t-1, t, t+1) is a legal move
            '''
            if self.arg.test_active_memory:
              right_state = utils.dense(y,
                                        output_dim = int(self.arg.hidden_size / 2),
                                        name = 'right_state')
            else:
              kernel = tf.get_variable('kernel',
                                       shape = [3, y.shape.as_list()[-1], int(self.arg.hidden_size / 2)],
                                       dtype = tf.float32)
              if self.arg.unidirectional_encoder:
                padding = 'VALID'
                y = tf.concat([tf.zeros([self.batch_size, 
                                         2, 
                                         self.arg.hidden_size]), 
                               y],
                              axis = 1)
              else:
                padding = 'SAME'
              right_state = tf.nn.convolution(y,
                                              kernel,
                                              padding = padding,
                                              name = 'convolution_conv_3x1')
            if self.arg.use_relu:
              right_state = tf.nn.relu(right_state)
            else:
              right_state = utils.gelu(right_state)
            right_state = self.dropout_fn(right_state)
          right_state = tf.pad(right_state,
                               [[0, 0], [0, 0], [0, int(self.arg.hidden_size * 4) - int(self.arg.hidden_size / 2)]],
                               constant_values = 0)
          y = left_state + right_state
          y = utils.layer_norm(y)
          if self.arg.test_active_memory:
            y = utils.dense(y,
                            output_dim = self.arg.hidden_size / 2,
                            name = 'separable_9x1')
          else:
            if self.arg.unidirectional_encoder:
              padding = 'VALID'
              y = tf.concat([tf.zeros([self.batch_size, 
                                       8, 
                                       self.arg.hidden_size * 4]), 
                             y],
                            axis = 1)
            else:
              padding = 'SAME'
            y = utils.separable_conv(y,
                                     filters = int(self.arg.hidden_size / 2),
                                     kernel_size = 9,
                                     padding = padding,
                                     name = 'separable_9x1')
          y = tf.pad(y,
                     [[0, 0], [0, 0], [0, int(self.arg.hidden_size/2)]],
                     constant_values = 0)
          x += self.dropout_fn(y)
        with tf.variable_scope('self_attention'):
          y = utils.layer_norm(x)
          y = utils.multihead_attention(query = y,
                                        memory = None,
                                        bias = self.encoder_self_attention_bias,
                                        total_key_depth = self.arg.head_size * self.arg.num_heads,
                                        total_value_depth = self.arg.head_size * self.arg.num_heads,
                                        output_depth = self.arg.hidden_size,
                                        num_heads = self.arg.num_heads,
                                        dropout_keep_prob = self.keep_prob,
                                        dropout_type = self.arg.dropout_type,
                                        relative_attention = self.arg.relative_attention,
                                        max_relative_position = self.arg.max_relative_position,
                                        adaptive_mask = self.arg.adaptive_mask)
          if self.arg.adaptive_mask:
            self.encoder_l0.append(y[1])
            y = y[0]
          x += self.dropout_fn(y)
        with tf.variable_scope('dense_layers'):
          y = utils.layer_norm(x)
          y = utils.dense(y,
                          output_dim = int(self.arg.hidden_size * 4),
                          name = 'dense_1')
          if self.arg.use_relu:
            y = tf.nn.relu(y)
          else:
            y = utils.gelu(y)
          y = self.dropout_fn(y)
          y = utils.dense(y,
                          output_dim = int(self.arg.hidden_size),
                          name = 'dense_2')
          x += self.dropout_fn(y)
    return utils.layer_norm(x)
    
  def decoder(self, inputs,
              memory,
              decoder_self_attention_bias,
              encoder_decoder_attention_bias):
    x = inputs
    if self.arg.adaptive_mask:
      self.decoder_l0 = []
    for layer in range(1,
                       self.arg.decoder_layers + 1):
      with tf.variable_scope('layer_{}'.format(layer)):
        with tf.variable_scope('16_head_self_attention'):
          y = utils.layer_norm(x)
          left_state = utils.multihead_attention(query = y,
                                                 memory = None,
                                                 bias = self.decoder_self_attention_bias,
                                                 total_key_depth = self.arg.head_size * max(min(self.arg.num_heads * 2,
                                                                                                16), 
                                                                                            self.arg.num_heads),
                                                 total_value_depth = self.arg.head_size * max(min(self.arg.num_heads * 2,
                                                                                                  16), 
                                                                                              self.arg.num_heads),
                                                 output_depth = self.arg.hidden_size,
                                                 num_heads = max(min(self.arg.num_heads * 2,
                                                                     16), 
                                                                 self.arg.num_heads),
                                                 dropout_keep_prob = self.keep_prob,
                                                 dropout_type = self.arg.dropout_type,
                                                 name = 'self_attention',
                                                 relative_attention = self.arg.relative_attention,
                                                 max_relative_position = self.arg.max_relative_position,
                                                 adaptive_mask = self.arg.adaptive_mask)
          if self.arg.adaptive_mask:
            self.decoder_l0.append(left_state[1])
            left_state = left_state[0]
          right_state = utils.multihead_attention(query = y,
                                                  memory = memory,
                                                  bias = self.encoder_decoder_attention_bias,
                                                  total_key_depth = self.arg.head_size * self.arg.num_heads,
                                                  total_value_depth = self.arg.head_size * self.arg.num_heads,
                                                  output_depth = self.arg.hidden_size,
                                                  num_heads = self.arg.num_heads,
                                                  dropout_keep_prob = self.keep_prob,
                                                  dropout_type = self.arg.dropout_type,
                                                  name = 'encoder_attention',
                                                  relative_attention = False,
                                                  max_relative_position = self.arg.max_relative_position,
                                                  adaptive_mask = self.arg.adaptive_mask)
          if self.arg.adaptive_mask:
            self.decoder_l0.append(right_state[1])
            right_state = right_state[0]
          x += self.dropout_fn(left_state) + self.dropout_fn(right_state)
        with tf.variable_scope('conv_branches'):
          y = utils.layer_norm(x)
          if self.arg.unidirectional_decoder:
            left_state = tf.concat([tf.zeros([self.batch_size, 
                                              10, 
                                              self.arg.hidden_size]), 
                                    y],
                                   axis = 1)
            padding = 'VALID'
          else:
            padding = 'SAME'
            left_state = y
          left_state = utils.separable_conv(left_state,
                                            filters = self.arg.hidden_size * 2,
                                            kernel_size = 11,
                                            padding = padding,
                                            name = 'separable_11x1')
          if self.arg.use_relu:
            left_state = tf.nn.relu(left_state)
          else:
            left_state = utils.gelu(left_state)
          left_state = self.dropout_fn(left_state)
          if self.arg.unidirectional_decoder:
            right_state = tf.concat([tf.zeros([self.batch_size, 
                                               6, 
                                               self.arg.hidden_size]), 
                                     y],
                                    axis = 1)
            padding = 'VALID'
          else:
            padding = 'SAME'
            right_state = y
          right_state = utils.separable_conv(right_state,
                                             filters = int(self.arg.hidden_size / 2),
                                             kernel_size = 7,
                                             padding = padding,
                                             name = 'separable_7x1')
          right_state = tf.pad(right_state,
                               paddings = [[0, 0], [0, 0], [0, int(self.arg.hidden_size * 1.5)]],
                               constant_values = 0)
          y = left_state + right_state
          y = utils.layer_norm(y)
          if self.arg.unidirectional_decoder:
            y = tf.concat([tf.zeros([self.batch_size, 
                                     6, 
                                     self.arg.hidden_size * 2]),
                           y],
                          axis = 1)
            padding = 'VALID'
          else:
            padding = 'SAME'
          y = utils.separable_conv(y,
                                   filters = self.arg.hidden_size,
                                   kernel_size = 7,
                                   padding = padding,
                                   name = 'separable_7x1_2')
          x += self.dropout_fn(y)
        with tf.variable_scope('self_attention'):
          y = utils.layer_norm(x)
          y = utils.multihead_attention(query = y,
                                        memory = None,
                                        bias = self.decoder_self_attention_bias,
                                        total_key_depth = self.arg.head_size * self.arg.num_heads,
                                        total_value_depth = self.arg.head_size * self.arg.num_heads,
                                        output_depth = self.arg.hidden_size,
                                        num_heads = self.arg.num_heads,
                                        dropout_keep_prob = self.keep_prob,
                                        dropout_type = self.arg.dropout_type,
                                        relative_attention = self.arg.relative_attention,
                                        max_relative_position = self.arg.max_relative_position,
                                        adaptive_mask = self.arg.adaptive_mask)
          if self.arg.adaptive_mask:
            self.decoder_l0.append(y[1])
            y = y[0]
          x += self.dropout_fn(y)
        with tf.variable_scope('encoder_attention'):
          y = utils.layer_norm(x)
          y = utils.multihead_attention(query = y,
                                        memory = memory,
                                        bias = self.encoder_decoder_attention_bias,
                                        total_key_depth = self.arg.head_size * self.arg.num_heads,
                                        total_value_depth = self.arg.head_size * self.arg.num_heads,
                                        output_depth = self.arg.hidden_size,
                                        num_heads = self.arg.num_heads,
                                        dropout_keep_prob = self.keep_prob,
                                        dropout_type = self.arg.dropout_type,
                                        relative_attention = False,
                                        max_relative_position = self.arg.max_relative_position,
                                        adaptive_mask = self.arg.adaptive_mask)
          if self.arg.adaptive_mask:
            self.decoder_l0.append(y[1])
            y = y[0]
          x += self.dropout_fn(y)
        with tf.variable_scope('dense_layers'):
          y = utils.layer_norm(x)
          y = utils.dense(y,
                          output_dim = self.arg.hidden_size * 4,
                          name = 'dense_1')
          y = tf.nn.swish(y)
          y = utils.layer_norm(y)
          y = utils.dense(y,
                          output_dim = self.arg.hidden_size,
                          name = 'dense_2')
          x += self.dropout_fn(y)
    return utils.layer_norm(x)
  
  def dropout_fn(self, x,
                 keep_prob = None):
    return tf.cond(self.training,
                   lambda: utils.dropout(x,
                                         keep_prob = self.keep_prob,
                                         dropout = self.arg.dropout_type),
                   lambda: tf.identity(x))
  
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
  
def argument():
  arg = optimize.argument()
  arg.dropout_type = 'vanilla' # 'vanilla'', 'alpha'
  arg.loss = 'sparse_softmax_cross_entropy_with_logits'
  arg.pos = 'linear_stop' # 'timing' 'emb' 'linear_stop' 'tanh_stop' 'exp_stop'
  
  arg.decoder_layers = 4 # the number of decoder layers
  arg.encoder_layers = 4 # the number of encoder layers
  arg.filter_size = 1024 # the filter size
  arg.head_size = 64 # the size of each head in the attention mechanisms
  arg.hidden_size = 256 # the hidden size
  arg.input_max_length = 10 # the maximum sequence size of the input, for the 'emb' pos
  arg.input_vocab_size = 1000 # the vocab size for the input
  arg.label_smoothing = 1.0 # the hyperparameter for label smoothing
  arg.max_relative_position = 100 # max relative position for relative attention
  arg.num_heads = 8 # the number of heads for the attention mechanisms
  arg.target_max_length = 10 # the maximum sequence size of the output, for the 'emb' pos
  arg.target_vocab_size = 1000 # the vocab size for the targets
  arg.weight_decay_hyperparameter = 0.001 # the hyperparameter for weight decay
  
  arg.adaptive_mask = False # whether adaptive mask is used
  arg.classification = False # whether the final output is a sequence, or single label
  arg.deparameterize = False # KEEP AS FALSE
  arg.dynamic_attention_span = False # KEEP AS FALSE
  arg.mask_loss = False # whether parts of the loss is masked
  arg.relative_attention = False # whether to use relative attention
  arg.unidirectional_decoder = True # whether the decoder is unidirectional
  arg.unidirectional_encoder = False # whether the encoder is unidirectional
  arg.use_decoder = True # whether to use the decoder
  arg.use_relu = True # whether the activation functions are ReLU or GELU
  arg.weight_decay_regularization = False # whether to use weight decay
  
  arg.test_active_memory = False
  
  return arg
  
if __name__ == '__main__':
  arg = argument()
  
  model = Evolved_Transformer(arg)