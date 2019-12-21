import numpy as np
import tensorflow as tf
import functools

import util_code as utils
import loss
import optimize
import mos
import develop_bias

class Transformer():
  def __init__(self, arg,
               name = None):
    
    '''
    the Transformer model, introduced in arXiv:1706.03762
    based off the code in tensor2tensor
    '''
    if name:
      self.name = name
    else:
      self.name = 'Transformer'
    batch_size = 128
    input_sequence_size = 27
    output_sequence_size = 27
    if __name__ != '__main__':
      batch_size = input_sequence_size = output_sequence_size = None
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
                                    shape = [batch_size, output_sequence_size], # (batch_size, output_sequence_size)
                                    name = 'targets')
    self.training = tf.placeholder(tf.bool)
    self.keep_prob = tf.placeholder(tf.float32)
    self.learning_rate = tf.placeholder(tf.float32)
    self.batch_size = tf.shape(self.inputs)[0]
    self.input_sequence_size = tf.shape(self.inputs)[1]
    if not self.arg.classification:
      self.target_sequence_size = tf.shape(self.targets)[1]
    
    self.encoder_self_attention_bias = develop_bias._create_mask(self.input_sequence_size,
                                                                 self.arg.unidirectional_encoder)
    if not self.arg.classification:
      self.encoder_decoder_attention_bias = tf.zeros([1, 1, self.target_sequence_size, self.input_sequence_size],
                                                     name = 'encoder_self_attention_bias')
      self.decoder_self_attention_bias = develop_bias._create_mask(self.target_sequence_size,
                                                                   self.arg.unidirectional_decoder)
    
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
    if not self.arg.classification:
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
    else:
      params = enc_params
      
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
      if self.arg.adaptive_mask:
        self.encoder_l0 = tf.reduce_sum(self.encoder_l0)
    if arg.use_decoder:
      with tf.variable_scope('decoder'):
        decoder_input = tf.pad(decoder_input,
                               paddings = [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        decoder_input = self.dropout_fn(decoder_input)
        decoder_output = self.decoder(decoder_input,
                                      encoder_output,
                                      decoder_self_attention_bias = self.decoder_self_attention_bias,
                                      encoder_decoder_attention_bias = self.encoder_decoder_attention_bias)
    if self.arg.classification:
      if self.arg.use_decoder:
        decoder_output = decoder_output[:,-1]
      else:
        encoder_output = encoder_output[:,-1]
    with tf.variable_scope('output'):
      if self.arg.use_mos:
        if self.arg.use_decoder:
          self.logits = mos.MoS(decoder_output,
                                hidden_size = self.arg.hidden_size,
                                vocab_size = self.arg.target_vocab_size)
        else:
          self.logits = mos.MoS(encoder_output,
                                hidden_size = self.arg.hidden_size,
                                vocab_size = self.arg.target_vocab_size)
        self.logits = tf.nn.softmax(self.logits)
        if self.arg.loss == 'sparse_softmax_cross_entropy_with_logits':
          self.arg.loss = 'log_loss'
        self.loss_cl = loss.Loss(self.logits,
                                 self.targets,
                                 self.arg.loss,
                                 vocab_size = self.arg.target_vocab_size,
                                 activation = tf.identity,
                                 label_smoothing = self.arg.label_smoothing)
        cost = tf.reduce_sum(self.loss_cl.loss,
                             axis = -1)
      else:
        weights = tf.get_variable('weights',
                                  shape = [self.arg.hidden_size, self.arg.target_vocab_size],
                                  dtype = tf.float32)
        bias = tf.get_variable('bias',
                               shape = [self.arg.target_vocab_size],
                               dtype = tf.float32)
        if arg.use_decoder:
          self.logits = tf.tensordot(decoder_output,
                                     weights,
                                     axes = 1) + bias
        else:
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
    if self.arg.adaptive_mask:
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
    x = inputs
    if self.arg.adaptive_mask:
      self.encoder_l0 = []
    for layer in range(1,
                       self.arg.encoder_layers + 1):
      with tf.variable_scope('layer_{}'.format(layer)):
        with tf.variable_scope('attention'):
          y = utils.layer_norm(x)
          y = utils.multihead_attention(query = y,
                                        memory = None,
                                        bias = encoder_self_attention_bias,
                                        total_key_depth = self.arg.head_size * self.arg.num_heads,
                                        total_value_depth = self.arg.head_size * self.arg.num_heads,
                                        output_depth = self.arg.hidden_size,
                                        num_heads = self.arg.num_heads,
                                        deparameterize = self.arg.deparameterize,
                                        dropout_keep_prob = self.keep_prob,
                                        dropout_type = self.arg.dropout_type,
                                        relative_attention = self.arg.relative_attention,
                                        max_relative_position = self.arg.max_relative_position,
                                        adaptive_mask = self.arg.adaptive_mask,
                                        dynamic_attention_span = self.arg.dynamic_attention_span)
          if self.arg.adaptive_mask:
            self.encoder_l0.append(y[1])
            y = y[0]
          y = self.dropout_fn(y)
          x += y
        
        with tf.variable_scope('ffd'):
          y = utils.layer_norm(x)
          y = self.ffd(y)
          y = self.dropout_fn(y)
          x += y
    with tf.variable_scope('output'):
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
        with tf.variable_scope('self_attention'):
          y = utils.layer_norm(x)
          y = utils.multihead_attention(query = y,
                                        memory = None,
                                        bias = decoder_self_attention_bias,
                                        total_key_depth = self.arg.head_size * self.arg.num_heads,
                                        total_value_depth = self.arg.head_size * self.arg.num_heads,
                                        output_depth = self.arg.hidden_size,
                                        num_heads = self.arg.num_heads,
                                        dropout_keep_prob = self.keep_prob,
                                        dropout_type = self.arg.dropout_type,
                                        relative_attention = self.arg.relative_attention,
                                        max_relative_position = self.arg.max_relative_position,
                                        adaptive_mask = self.arg.adaptive_mask,
                                        dynamic_attention_span = self.arg.dynamic_attention_span)
          if self.arg.adaptive_mask:
            self.decoder_l0.append(y[1])
            y = y[0]
          y = self.dropout_fn(y)
          x += y
        with tf.variable_scope('encoder_attention'):
          y = utils.layer_norm(x)
          y = utils.multihead_attention(query = y,
                                        memory = memory,
                                        bias = encoder_decoder_attention_bias,
                                        total_key_depth = self.arg.head_size * self.arg.num_heads,
                                        total_value_depth = self.arg.head_size * self.arg.num_heads,
                                        output_depth = self.arg.hidden_size,
                                        num_heads = self.arg.num_heads,
                                        dropout_keep_prob = self.keep_prob,
                                        dropout_type = self.arg.dropout_type,
                                        relative_attention = False,
                                        max_relative_position = self.arg.max_relative_position,
                                        adaptive_mask = self.arg.adaptive_mask,
                                        dynamic_attention_span = self.arg.dynamic_attention_span)
          if self.arg.adaptive_mask:
            self.decoder_l0.append(y[1])
            y = y[0]
          y = self.dropout_fn(y)
          x += y
        with tf.variable_scope('ffd'):
          y = utils.layer_norm(x)
          y = self.ffd(y)
          y = self.dropout_fn(y)
          x += y
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
    '''
    the timing positions include:
    - timing_signal, based off tensor2tensor
    - emb signal, based off tensor2tensor
    - linear_stop, based off arXiv:1804.00964
    - tanh_stop, based off arXiv:1804.00964
    - exp_stop, based off arXiv:1804.00964
    '''
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
  arg.dropout_type = 'vanilla' # 'vanilla', 'broadcast', 'alpha'
  arg.ffd = 'transformer_ffd' # 'transformer_ffd' 'sru' 'sepconv'
  arg.loss = 'sparse_softmax_cross_entropy_with_logits'
  arg.pos = 'timing' # 'timing' 'emb' 'linear_stop' 'tanh_stop' 'exp_stop'
  
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
  arg.use_mos = False # whether to use an MoS
  arg.use_relu = True # whether the activation functions are ReLU or GELU
  arg.weight_decay_regularization = False # whether to use weight decay
  return arg
  
if __name__ == '__main__':
  arg = argument()
  
  model = Transformer(arg)