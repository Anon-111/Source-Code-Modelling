import math
import numpy as np
import tensorflow as tf

def log(x):
  '''
  because there is a lower-bound, the log-function is numerically stable
  x - a tensor, array, float or int
  '''
  if type(x) in [int, float, np.ndarray]:
    return math.log(np.maximum(x,
                               1e-8))
  else:
    return tf.log(tf.maximum(x,
                             1e-8))
  
def add_timing_signal_1d(sequence_size,
                         channels,
                         log_fn = log,
                         min_timescale = 1.0,
                         max_timescale = 1.0e4,
                         start_index = 0):
  '''
  used as a timing-embedding vector for Transformer networks
  function initially taken from tensor2tensor function, get_timing_signal_1d
  example found at https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
  sequence_size - the middle dimension of the tensor, given that the tensor to be embedded is of shape [batch_size, sequence_size, hidden_dim]
  channels - hidden_dim
  '''
  position = tf.cast(tf.range(sequence_size) + start_index,
                     dtype = tf.float32)
  num_timescales = channels // 2
  log_timescale_increment = (log_fn(float(max_timescale) / float(min_timescale)) / tf.maximum(tf.cast(num_timescales,
                                                                                                      dtype = tf.float32) - 1,
                                                                                              1))
  inv_timescales = min_timescale * tf.exp(tf.cast(tf.range(num_timescales),
                                                  dtype = tf.float32) * - log_timescale_increment)
  scaled_time = tf.expand_dims(position,
                               axis = 1) * tf.expand_dims(inv_timescales,
                                                          axis = 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)],
                     axis = 1)
  try:
    signal = tf.pad(signal,
                    [[0, 0], [0, tf.math.mod(channels, #tf.math.mod
                                             2)]])
  except:
    signal = tf.pad(signal,
                    [[0, 0], [0, tf.mod(channels, #tf.math.mod
                                             2)]])
  return tf.reshape(signal,
                    [1, sequence_size, channels])
    
def add_positional_embedding(inputs,
                             max_length,
                             hidden_size,
                             input_sequence_size,
                             name = None):
  '''
  used as a timing-embedding vector for Transformer networks
  function initially taken from tensor2tensor function, add_positional_embedding
  example found at https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
  inputs - the tensor to be embedded, dtype = tf.float32 shape = [batch_size, sequence_size, hidden_size]
  max_length - the maximum size that sequence_size could be
  hidden_size - hidden_size
  input_sequence_size - sequence_size
  name - the scope of the operator
  '''
  length = tf.shape(inputs)[1]
  with tf.variable_scope(name,
                         default_name = 'positional_embedding'):
    var = tf.get_variable('positional_variables',
                          shape = [max_length, hidden_size],
                          dtype = inputs.dtype)
    pad_length = tf.maximum(0,
                            input_sequence_size - max_length)
    sliced = tf.cond(tf.less(length,
                             max_length),
                     lambda: tf.slice(var,
                                      [0, 0],
                                      [input_sequence_size, -1]),
                     lambda: tf.pad(var,
                                    [[0, pad_length],
                                     [0, 0]]))
    return tf.expand_dims(sliced,
                          axis = 0)
  
def layer_norm(inputs,
               filters = None,
               epsilon = 1e-6,
               name = None,
               reuse = None,
               dtype = None):
  '''
  layer normalization function
  inputs - a tensor
  filters - the size of the final dimension of inputs
  epsilon - a hyperparameter used for numerical stability
  name - the scope of the operator
  reuse - whether to reuse the scope
  dtype - the dtype of inputs
  '''
  if filters is None:
    filters = inputs.get_shape().as_list()[-1]
  if dtype == None:
    dtype = inputs.dtype
  with tf.variable_scope(name,
                         default_name = 'layer_norm',
                         reuse = reuse):
    scale = tf.get_variable('scale',
                            shape = [filters],
                            initializer = tf.ones_initializer(),
                            dtype = dtype)
    bias = tf.get_variable('bias',
                           shape = [filters],
                           initializer = tf.zeros_initializer(),
                           dtype = dtype)
    mean = tf.reduce_mean(inputs,
                          axis = [-1],
                          keepdims = True)
    try:
      variance = tf.reduce_mean(tf.math.squared_difference(inputs, 
                                                           mean),
                                axis = [-1],
                                keepdims = True)
      return (inputs - mean) * tf.math.rsqrt(variance + epsilon) * scale + bias # tf.math.rsqrt
    except:
      variance = tf.reduce_mean(tf.squared_difference(inputs, # tf.math.squared_difference
                                                           mean),
                                axis = [-1],
                                keepdims = True)
      return (inputs - mean) * tf.rsqrt(variance + epsilon) * scale + bias # tf.math.rsqrt
    
def dense(inputs,
          input_dim = None,
          output_dim = None,
          use_bias = True,
          weight_initialization = tf.random_normal_initializer(mean = 0.0,
                                                               stddev = 1.0),
          bias_initialization = tf.zeros_initializer(),
          dtype = None,
          name = None,
          reuse = False):
  '''
  feedforward layer
  inputs - a tensor, shape = [..., hidden_size]
  input_dim - int, hidden_size
  output_dim - int, the size of the output for the feedforward layer
  use_bias - whether to add a trainable bias vector
  weight_initialization - the initialization of the weights. Default set to N(0, 1)
  bias_initialization - the initialization of the bias. Defauly set to 0
  dtype - the dtype of the input tensor
  name - the scope of the operator
  reuse - whether to reuse a previous feedforward layer
  '''
  if dtype == None:
    dtype = inputs.dtype
  with tf.variable_scope(name,
                         default_name = 'dense',
                         reuse = reuse):
    if input_dim == None:
      input_dim = inputs.get_shape().as_list()[-1]
    if output_dim == None:
      output_dim = input_dim
    weights = tf.get_variable('weights',
                              shape = [input_dim, output_dim],
                              dtype = dtype,
                              initializer = weight_initialization,
                              trainable = True)
    if use_bias:
      bias = tf.get_variable('bias',
                             shape = [output_dim],
                             dtype = dtype,
                             initializer = bias_initialization,
                             trainable = True)
      return tf.tensordot(inputs,
                          weights,
                          axes = 1) + bias
    else:
      return tf.tensordot(inputs,
                          weights,
                          axes = 1)
    
def alpha_dropout(inputs,
                  keep_prob,
                  alpha = -1.7580993408473766,
                  fixedPointMean = 0.0,
                  fixedPointVar = 1.0,
                  noise_shape = None,
                  seed = None,
                  name = None):
  '''
  the dropout used alongside the SeLU activation function
  function was entirely written by https://github.com/bioinf-jku/SNNs/blob/master/selu.py
  inputs - the tensor to be subject to dropout
  keep_prob - the probability of keeping an element
  '''
  def dropout_selu_impl(x,
                        keep_prob,
                        alpha,
                        noise_shape,
                        seed):
    x = tf.convert_to_tensor(x,
                             name = 'x')
    alpha = tf.convert_to_tensor(alpha,
                                 dtype = x.dtype,
                                 name = 'alpha')
    if noise_shape == None:
      noise_shape = tf.shape(x)
    random_tensor = keep_prob + tf.random.uniform(noise_shape,
                                                  seed = seed,
                                                  dtype = x.dtype)
    binary_tensor = tf.floor(random_tensor)
    ret = x * binary_tensor + alpha * (1 - binary_tensor)
    a = tf.sqrt(fixedPointVar / (keep_prob * ((1 - keep_prob) * tf.pow(alpha - fixedPointMean,
                                                                       2) + fixedPointVar)))
    b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
    ret = a * ret + b
    ret.set_shape(x.get_shape())
    return ret
  with tf.variable_scope(name,
                         default_name = 'alpha_dropout'):
    return dropout_selu_impl(inputs,
                             keep_prob,
                             alpha,
                             noise_shape,
                             seed)

def dropout(inputs,
            keep_prob,
            dropout = 'vanilla'):
  '''
  over-riding dropout function, that can call either of the primary dropout functions
  vanilla dropout functions is the most common form of dropout across literature
  inputs - the tensor
  keep_prob - the probability of every element in inputs to be kept after dropout
  dropout - type of dropout, 'vanilla', 'alpha'
  '''
  assert dropout == 'vanilla' or dropout == 'alpha', '{} is not a valid dropout algorithm'.format(dropout)
  if dropout == 'vanilla':
    if tf.__version__ == '1.14.0':
      return tf.nn.dropout(inputs,
                           rate = 1 - keep_prob)
    else:
      return tf.nn.dropout(inputs,
                           keep_prob = keep_prob) #rate = 1 - keep_prob)
  else:
    return alpha_dropout(inputs,
                         keep_prob = keep_prob)

def compute_attention_component(inputs,
                                total_depth,
                                filter_width = 1,
                                padding = 'VALID',
                                weight_initialization = tf.initializers.random_normal(stddev = 0.1),
                                name = 'c',
                                reuse = False,
                                stop_gradient = False):
  if filter_width == 1:
    with tf.variable_scope(name,
                           default_name = 'compute_attention_component',
                           reuse = reuse):
      return dense(inputs,
                   output_dim = total_depth,
                   use_bias = False,
                   weight_initialization = weight_initialization,
                   name = name)
  else:
    with tf.variable_scope(name,
                           default_name = 'conv1d',
                           reuse = reuse):
      kernel = tf.get_variable('kernel',
                               shape = [filter_width, inputs.shape.as_list()[-1], total_depth],
                               dtype = tf.float32)
      return tf.nn.convolution(inputs,
                               kernel,
                               padding = padding)
                             
def compute_qkv(query,
                memory,
                total_key_depth,
                total_value_depth,
                deparameterize = False,
                q_filter_width = 1,
                kv_filter_width = 1,
                q_padding = 'VALID',
                kv_padding = 'VALID'):
  if memory is None:
    memory = query
  if deparameterize:
    scope = ['q', 'q', 'q']
  else:
    scope = ['q', 'k', 'v']
  q = compute_attention_component(query,
                                  total_depth = total_key_depth,
                                  filter_width = q_filter_width,
                                  padding = q_padding,
                                  name = scope[0])
  k = compute_attention_component(memory,
                                  total_depth = total_key_depth,
                                  filter_width = kv_filter_width,
                                  padding = kv_padding,
                                  name = scope[1],
                                  reuse = (scope[1] == 'q'))
  v = compute_attention_component(memory,
                                  total_depth = total_value_depth,
                                  filter_width = kv_filter_width,
                                  padding = kv_padding,
                                  name = scope[2],
                                  reuse = (scope[2] == 'q'))
  return q, k, v

def split_heads(inputs,
                n):
  hidden_size = inputs.shape.as_list()[-1]
  x_shape = tf.shape(inputs)
  length = tf.shape(inputs).shape.as_list()[0] - 1
  shape = []
  for i in range(length):
    shape.append(x_shape[i])
  shape.extend([n, hidden_size // n])
  return tf.transpose(tf.reshape(inputs,
                                 shape),
                      [0, 2, 1, 3])

def combine_heads(x):
  x = tf.transpose(x,
                   [0, 2, 1, 3])
  x_shape = tf.shape(x)
  length = tf.shape(x).shape.as_list()[0] - 2
  shape = []
  for i in range(length):
    shape.append(x_shape[i])
  shape.append(x_shape[-2] * x_shape[-1])
  return tf.reshape(x,
                    shape = shape)

def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_keep_prob,
                          dropout_type,
                          adaptive_mask = False,
                          dynamic_attention_span = False,
                          name = None):
  with tf.variable_scope(name,
                         default_name = 'dot_product_attention'):
    logits = tf.matmul(q, 
                       k,
                       transpose_b = True)
    logits /= k.shape.as_list()[-1]
    if bias is not None:
      logits += bias
    with tf.variable_scope('attention_weights'):
      weights = tf.nn.softmax(logits,
                              name = 'attention_weights')
      
      if adaptive_mask:
        weights, mask_loss = adaptive_span(weights,
                                           dynamic_attention_span = dynamic_attention_span)
      else:
        mask_loss = None
    weights = dropout(weights,
                      keep_prob = dropout_keep_prob,
                      dropout = dropout_type)
    return tf.matmul(weights,
                     v), mask_loss

def _generate_relative_positions_matrix(length_q,
                                        length_k,
                                        max_relative_position):
  range_vec_k = tf.range(length_k)
  range_vec_q = range_vec_k[-length_q:]
  distance_mat = range_vec_k[None, :] + range_vec_q[:, None]
  distance_mat_clipped = tf.clip_by_value(distance_mat,
                                          -max_relative_position,
                                          max_relative_position)
  return distance_mat_clipped + max_relative_position
  
def _generate_relative_positions_embeddings(length_q,
                                            length_k,
                                            depth,
                                            max_relative_position,
                                            name):
  with tf.variable_scope(name):
    relative_positions_matrix = _generate_relative_positions_matrix(length_q,
                                                                    length_k,
                                                                    max_relative_position)
    vocab_size = max_relative_position * 2 + 1
    embeddings_table = tf.get_variable('embeddings',
                                       [vocab_size, depth])
    embeddings = tf.gather(embeddings_table,
                           relative_positions_matrix)
  return embeddings

def _relative_attention_inner(x,
                              y,
                              z,
                              transpose):
  batch_size = tf.shape(x)[0]
  heads = x.shape.as_list()[1]
  sequence_size = tf.shape(x)[2]
  xy_matmul = tf.matmul(x,
                        y,
                        transpose_b = transpose)
  x_t = tf.transpose(x,
                     [2, 0, 1, 3])
  x_t_r = tf.reshape(x_t,
                     [sequence_size, heads * batch_size, -1])
  x_tz_matmul = tf.matmul(x_t_r,
                          z,
                          transpose_b = transpose)
  x_tz_matmul_r = tf.reshape(x_tz_matmul,
                             [sequence_size, batch_size, heads, -1])
  x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r
                                 [1, 2, 0, 3])
  return xy_matmul + x_tz_matmul_r_t
  
def relative_dot_product_attention(q,
                                   k,
                                   v,
                                   bias,
                                   max_relative_position,
                                   dropout_keep_prob,
                                   dropout_type,
                                   adaptive_mask = False,
                                   dynamic_attention_span = False,
                                   name = None):
  with tf.variable_scope(name,
                         default_name = 'relative_dot_product'):
    relations_keys = _generate_relative_positions_embeddings(tf.shape(q)[2],
                                                             tf.shape(k)[2],
                                                             q.shape.as_list()[-1],
                                                             max_relative_position,
                                                             name = 'keys')
    relations_values = _generate_relative_positions_embeddings(tf.shape(q)[2],
                                                               tf.shape(k)[2],
                                                               q.shape.as_list()[-1],
                                                               max_relative_position,
                                                               name = 'values')
    logits = _relative_attention_inner(q,
                                       k,
                                       relations_keys,
                                       True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits,
                            name = 'attention_weights')
    
    if adaptive_mask:
      weights, mask_loss = adaptive_span(weights,
                                         dynamic_attention_span = dynamic_attention_span)
    else:
      mask_loss = None
    
    weights = dropout(weights,
                      keep_prob = dropout_keep_prob,
                      dropout = dropout_type)
    return _relative_attention_inner(weights,
                                     v,
                                     relations_values,
                                     False), mask_loss

def multihead_attention(query,
                        memory,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_keep_prob,
                        dropout_type = 'vanilla',
                        deparameterize = False,
                        relative_attention = False,
                        max_relative_position = 100,
                        adaptive_mask = False,
                        dynamic_attention_span = False,
                        name = None):
  '''
  a multihead attention operator. Based heavily of tensor2tensor
  query - a tensor, dtype = tf.float32 shape = [batch_size, query_sequence_size, hidden_size]
  memory - a tensor or None, dtype = tf.float32 shape = [batch_size, memory_sequence_size, hidden_size]. If None, memory is specified as query and is self-attention
  bias - the bias of the multihead attention. See develop_bias.py
  total_key_depth - the size of the key's final dimension
  total_value_depth - the size of the value's final dimension
  output_depth - the size of the output vector's final dimension
  num_heads - the number of heads of the attention mechanism
  dropout_keep_prob - the probability of attention vector being kept after dropout
  dropout_type - either 'vanilla' or 'alpha'
  deparameterize - whether to deparameterize parts of the attention mechanism. Not as efficient. KEEP AS FALSE
  relative_attention - whether to use a relative attention mechanism
  max_relative_position - used in relative_attention
  adaptive_mask - whether to use adaptive_mask
  dynamic_attention_span - KEEP AS FALSE
  name - the scope of the operator
  '''
  with tf.variable_scope(name,
                         default_name = 'attention'):
    q, k, v = compute_qkv(query = query,
                          memory = memory,
                          total_key_depth = total_key_depth,
                          total_value_depth = total_value_depth,
                          deparameterize = deparameterize)
    q = split_heads(q,
                    num_heads)
    k = split_heads(k,
                    num_heads)
    v = split_heads(v,
                    num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head ** -0.5
    if relative_attention:
      x, mask_loss = relative_dot_product_attention(q,
                                                    k,
                                                    v,
                                                    bias,
                                                    max_relative_position,
                                                    dropout_keep_prob,
                                                    dropout_type,
                                                    adaptive_mask,
                                                    dynamic_attention_span)
    else:
      x, mask_loss = dot_product_attention(q,
                                           k,
                                           v,
                                           bias,
                                           dropout_keep_prob,
                                           dropout_type,
                                           adaptive_mask,
                                           dynamic_attention_span)
    x = combine_heads(x)
    x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])
    if adaptive_mask:
      return dense(x,
                   output_dim = output_depth,
                   use_bias = False,
                   name = 'output_transform'), mask_loss
    else:
      return dense(x,
                   output_dim = output_depth,
                   use_bias = False,
                   name = 'output_transform')
  
def adaptive_span(weights,
                  dynamic_attention_span = False,
                  init_val = 0.0,
                  ramp_size = 32):
  '''
  adaptive span is specified in arXiv:1905.07799
  Dynamic attention span is currently not functional.
  Called from multihead_attention
  '''
  head_size = weights.shape.as_list()[1]
  attn_span = tf.cast(tf.shape(weights)[-1],
                      dtype = tf.float32)
  
  with tf.variable_scope('adaptive_mask'):
    if dynamic_attention_span:
      print('DYNAMIC ATTENTION SPAN WONT WORK')
      exit()
      current_val = dense(tf.reshape(weights,
                                     [tf.shape(weights)[0], head_size, -1]),
                          output_dim = 1,
                          weight_initialization = tf.zeros_initializer(),
                          bias_initialization = tf.constant_initializer(-4),
                          use_bias = True)
      current_val = tf.expand_dims(current_val,
                                   axis = -1)
    else:
      current_val = tf.Variable(np.zeros(shape = [head_size, 1, 1]) + init_val,
                                name = 'current_val',
                                dtype = tf.float32)
    mask_template = tf.range(attn_span - 1,
                             -1,
                             delta = -1,
                             dtype = tf.float32)
    mask_template = tf.reshape(mask_template,
                               [1, 1, -1])
    mask = mask_template + current_val * attn_span
    mask /= (ramp_size + 1)
    mask = tf.clip_by_value(mask,
                            0,
                            1)
    
    weights = mask*weights
    
    weights /= (tf.reduce_sum(weights,
                              axis = -1,
                              keepdims = True) + 1e-8)
  l0 = tf.reduce_mean(current_val) * attn_span
  
  return weights, l0

def embedding(x,
              vocab_size,
              model_dim = None,
              embed_size = None,
              name = None,
              use_tpu = False,
              initializer = tf.initializers.random_normal(stddev = 1.0),
              lookup_table = None):
  '''
  an embedding function
  x - a tensor, dtype = tf.int32 shape = [...]
  vocab_size - the vocab_size of x
  embed_size - an int, so that x is initially embedded to embed_size and then sent through an FFD to model_dim size
  name - the scope of the operator
  initializer - the initializer of the embedding weights. Default is N(0, 1)
  '''
  embed_size = model_dim
  with tf.variable_scope(name,
                         default_name = 'embedding'):
    emb_scale = embed_size ** 0.5
    if lookup_table is None:
      lookup_table = tf.get_variable('lookup_table',
                                     shape = [vocab_size, embed_size],
                                     initializer = initializer)
    if use_tpu:
      x = tf.one_hot(x,
                     depth = vocab_size,
                     dtype = tf.float32)
      y = tf.matmul(x, 
                    lookup_table)
    else:
      y = tf.nn.embedding_lookup(lookup_table,
                                 x)
    return y * emb_scale, lookup_table
    
def separable_conv(inputs,
                   filters,
                   kernel_size,
                   strides = 1,
                   padding = 'VALID',
                   dilation_rate = 1,
                   depth_multiplier = 1,
                   activation = tf.identity,
                   use_bias = True,
                   name = None,
                   reuse = False):
  '''
  a depthwise separable convolutional operator. Relatively little practical testing. 
  Advise against use for the time being
  inputs - a tensor, dtype = tf.float32 shape = [batch_size, kernel_size, hidden_size] or [batch_size, kernel_one, kernel_two, hidden_size]
  filters - an int
  kernel_size - the kernel size of the operator
  strides - the strides of the operator
  padding = 'VALID' or 'SAME'
  dilation_rate
  depth_multiplier
  activation - the activation function applied to the final output
  use_bias - whether to add a trainable bias vector
  name - the scope of the operator
  reuse - whether to reuse the scope
  '''
  with tf.variable_scope(name,
                         default_name = 'separable_conv',
                         reuse = reuse):
    if type(strides) == int:
      strides = (strides, strides)
    if len(strides) == 1:
        strides = (strides[0], 
                   strides[0])
    strides = (1, ) + strides + (1, )
    if type(dilation_rate) == int:
      dilation_rate = (dilation_rate, )
    
    if len(inputs.shape.as_list()) == 4:
      rank = 2
    elif len(inputs.shape.as_list()) == 3:
      rank = 1
    else:
      print('Rank is {}, must be 3 or 4'.format(len(inputs.shape.as_list()) - 2))
      exit()
    if type(kernel_size) == int:
      if rank == 1:
        kernel_size = (kernel_size, )
      elif rank == 2:
        kernel_size = (kernel_size, kernel_size, )
      
    input_dim = inputs.shape.as_list()[-1]
    depthwise_kernel_shape = kernel_size + (input_dim, 
                                            depth_multiplier)
    pointwise_kernel_shape = (1, ) * rank + (depth_multiplier * input_dim, 
                                             filters)
    depthwise_kernel = tf.get_variable('depthwise_kernel',
                                       shape = depthwise_kernel_shape,
                                       dtype = tf.float32)
    pointwise_kernel = tf.get_variable('pointwise_kernel',
                                       shape = pointwise_kernel_shape,
                                       dtype = tf.float32)
    if use_bias:
      bias = tf.get_variable('bias',
                             shape = (filters, ),
                             dtype = tf.float32)
  
    if rank == 1:
      inputs = tf.expand_dims(inputs,
                              axis = 1)
      depthwise_kernel = tf.expand_dims(depthwise_kernel,
                                        axis = 0)
      pointwise_kernel = tf.expand_dims(pointwise_kernel,
                                        axis = 0)
      dilation_rate = (1, ) + dilation_rate
    
    output = tf.nn.separable_conv2d(inputs,
                                    depthwise_kernel,
                                    pointwise_kernel,
                                    strides = strides,
                                    padding = padding,
                                    rate = dilation_rate)
    if rank == 1:
      output = tf.squeeze(output,
                          axis = 1)
    if use_bias:
      output += bias
    return activation(output)
  
def convolution_gating(inputs,
                       kernel_size,
                       output_dim,
                       input_dim = None,
                       name = None):
  '''
  A convolutional gating layer, that returns conv(inputs) * sigmoid(conv(inputs))
  Initially introduced in arXiv:1612.08083
  inputs - a tensor, dtype = tf.float32 shape = [batch_size, sequence_size, hidden_size]
  kernel_size - an int
  output_dim - an int
  input_dim - an int
  name - the scope for the operator
  '''
  if input_dim:
    assert input_dim == inputs.shape.as_list()[-1]
  else:
    input_dim = inputs.shape.as_list()[-1]
  with tf.variable_scope(name,
                         default_name = 'gated_linear_unit'):
    with tf.variable_scope('linear'):
      d = tf.get_variable('weights',
                          shape = [kernel_size, input_dim, output_dim],
                          dtype = tf.float32)
      b = tf.get_variable('bias',
                          shape = [output_dim],
                          dtype = tf.float32)
      a = tf.nn.convolution(inputs,
                            filter = d,
                            padding = 'VALID') + b
    with tf.variable_scope('gating'):
      e = tf.get_variable('weights',
                          shape = [kernel_size, input_dim, output_dim],
                          dtype = tf.float32)
      c = tf.get_variable('bias',
                          shape = [output_dim],
                          dtype = tf.float32)
      f = tf.nn.convolution(inputs,
                            filter = e,
                            padding = 'VALID') + c
  return tf.multiply(a,
                     tf.nn.sigmoid(f))

def gelu(x):
  '''
  activation functions introduced in https://github.com/hendrycks/GELUs
  GELU was used to replace ReLU in BERT Transformer
  faster, but less accurate approximation can be defined as:
  sigmoid(1.702 * x) * x
  '''
  return 0.5 * x * (1 + tf.nn.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x,
                                                                               3))))

def squeeze_elu(x):
  return tf.where(x > 0,
                  1 - tf.exp(-x),
                  tf.exp(x) - 1)

def exprelu(x):
  # return tf.where(x > 0.0, log(x + 1), -1 * tf.log(-1 * x + 1))
  return tf.where(x > 1.0, 
                  tf.log(x) + 1, 
                  tf.exp(x - 1))

def reduce_var(x, axis = None, keepdims = False):
  m = tf.reduce_mean(x, axis = axis, keepdims = True)
  devs_squared = tf.square(x - m)
  return tf.reduce_mean(devs_squared, axis = axis, keepdims = keepdims)

def reduce_std(x, axis = None, keepdims = False):
  return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))