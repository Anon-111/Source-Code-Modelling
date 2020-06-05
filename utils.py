import math
import numpy as np
import tensorflow as tf

def gelu(x):
  # the GeLU activation function can be used in various Transformer layers
  # originally introduced in arXiv:1606.08415
  # the ReLU activation function in the Transformer model (arXiv:1706.03762) is replaced by GELU in BERT (arXiv:1810.04805)
  # a less accurate but faster approximation is 
  # tf.nn.sigmoid(1.702 * x) * x
  return 0.5 * x * (1 + tf.nn.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x,
                                                                               3))))

def log_fn(x):
  # numerically stabile logarithm function
  # as the input decreases, the output asymtopically approaches -np.inf
  # log(np.max(0, 1e-8)) = -18.42
  # log(0) = -np.inf
  if type(x) in [int, float, np.ndarray]:
    return math.log(np.maximum(x,
                               1e-8))
  else:
    return tf.math.log(tf.maximum(x,
                                  1e-8))

def sigmoid_cutoff(x):
  # a cutoff sigmoid function, as recommended by arXiv:1511.08228
  # sigmoid output is cutoff between (0.0833, 0.9166)
  # the gradients of the cutoff activation function is always between (0.07611, 0.25)
  # the gradients do not disappear to 0 at the extremes, allowing for a more stable training method
  # lower chance of disappearing gradients
  # other forms of sigmoid cutoffs have been suggested
  return tf.maximum(tf.minimum(1.2 * tf.sigmoid(x) - 0.1,
                               1.0),
                    0.0)
                    
class Active_Memory(tf.keras.layers.Layer):
  # the active-memory mechanisms, based on the Neural GPU arXiv:1511.08228
  # Active_Memory, in this function, initially in arXiv:1912.11959
  # this allows for a sequential analysis, where the entire analysis can be analyzed evenly
  def __init__(self, kernel_size,
               memory_mechanism,
               causal = True,
               activation = tf.nn.relu,
               output_multiplier = 1,
               dilation_rate = 1,
               persistant_vector = None,
               h = 16,
               output_size = None,
               **kwargs):
    # kernel_size - the size of the convolutional operator and, by extension, the the window size of the operator
    # given a larger kernel_size, the mechanism is capable of seeing across further time-steps
    # memory_mechanism - a string representing the chosen memory_mechanism. See below
    # causal - if True, the model is unidirectional (used for tasks such as language modelling)
    # if False, the model is bidirectional (used for tasks where the model should be capable of seeing across the entire sequence)
    # activation - the activation function to be used for the active-memory. The default is the ReLU activation function
    # output_multiplier - only used in 'depthwise_convolution'
    # dilation_rate - only used in 'dilated_convolution'
    # persistant_vector - only used in 'persistant_convolution'. the persistant_vector, if not provided to the class will be built
    # the persistant_vector must be of shape [1, kernel_size - 1, hidden_size]
    # h - the parameter of the tied weights for the lightweight and dynamic lightweight operators 
    # must divide the hidden_size of the input vector without remainder
    super(Active_Memory,
          self).__init__(**kwargs)
    self.kernel_size = kernel_size
    self.memory_mechanism = memory_mechanism
    self.causal = causal
    self.activation = activation
    self.output_multiplier = output_multiplier
    self.dilation_rate = dilation_rate
    self.persistant_vector = persistant_vector
    self.h = h
    self.output_size = output_size
    
    # assert that the memory_mechanism given to the class is recognized
    assert self.assert_mechanism(), '{} memory mechanism is not recognized'.format(self.memory_mechanism)
    
  def build(self, input_shape):
    # given an input vector of size [..., hidden_size], the output vector will be of size [..., hidden_size]
    self.hidden_size = input_shape.as_list()[-1]
    if self.output_size is None:
      self.output_size = self.hidden_size
    if self.memory_mechanism == 'convolution':
      # (kernel_size * hidden_size ** 2) parameters
      self.kernel = self.add_weight('kernel',
                                    shape = [self.kernel_size, self.hidden_size, self.output_size],
                                    dtype = tf.float32)
    elif self.memory_mechanism == 'highway_convolution' or self.memory_mechanism == 'output_convolution':
      # (2 * kernel_size * hidden_size ** 2) parameters
      self.kernel = self.add_weight('kernel',
                                    shape = [self.kernel_size, self.hidden_size, self.hidden_size],
                                    dtype = tf.float32)
      self.sigmoid_kernel = self.add_weight('sigmoid-kernel',
                                            shape = [self.kernel_size, self.hidden_size, self.hidden_size],
                                            dtype = tf.float32)
    elif self.memory_mechanism == 'depthwise_convolution':
      # (kernel_size * hidden_size * output_multiplier) parameters
      # for reasons described below, the shape is expanded across 0th axis
      # the output_multiplier is often set to 1
      self.kernel = self.add_weight('kernel',
                                    shape = [1, self.kernel_size, self.hidden_size, self.output_multiplier],
                                    dtype = tf.float32)
    elif 'depthwise_convolution' in self.memory_mechanism:
      # this contains both highway-depthwise and persistant-depthwise
      # I've haven't changed the outside functionality
      self.kernel = self.add_weight('kernel',
                                    shape = [1, self.kernel_size, self.hidden_size, self.output_multiplier],
                                    dtype = tf.float32)
      if 'highway' in self.memory_mechanism:
        self.sigmoid_kernel = self.add_weight('kernel',
                                              shape = [1, self.kernel_size, self.hidden_size, self.output_multiplier],
                                              dtype = tf.float32)
      if 'persistant' in self.memory_mechanism:
        self.persistant_vector = self.add_weight('persistant-vector',
                                                 shape = [1, self.kernel_size - 1, self.hidden_size],
                                                 dtype = tf.float32)
    elif self.memory_mechanism == 'persistant_convolution':
      # for the time being, only unidirectional analysis is permitted for persistant convolution
      assert self.causal
      if self.persistant_vector is None:
        # if persistant_vector is not presented to the class, it must be produced
        self.persistant_vector = self.add_weight('persistant-vector',
                                                 shape = [1, self.kernel_size - 1, self.hidden_size],
                                                 dtype = tf.float32)
      # (kernel_size * hidden_size ** 2 + (kernel_size - 1) * hidden_size) parameters
      self.kernel = self.add_weight('kernel',
                                    shape = [self.kernel_size, self.hidden_size, self.hidden_size],
                                    dtype = tf.float32)
    elif self.memory_mechanism == 'dilated_convolution':
      # the input at time-step t depends on t, t - dilation_rate, t - dilation_rate * 2, etc for kernel_size
      # the keras conv layer is used only for dilated convolution.
      # note that this is only memory_mechanism that uses self.dilation_rate
      self.layer = tf.keras.layers.Conv1D(filters = self.hidden_size,
                                          kernel_size = self.kernel_size,
                                          padding = 'causal' if self.causal else 'same',
                                          dilation_rate = self.dilation_rate,
                                          use_bias = False)
    elif self.memory_mechanism == 'dynamic_lightweight_conv':
      # the dynamic_lightweight_conv is implemented in another class
      self.layer = Dynamic_Lightweight_Convolution(kernel_size = self.kernel_size,
                                                   h = self.h,
                                                   use_dynamic_weights = True)
    elif self.memory_mechanism == 'lightweight_conv':
      # similar to above
      self.layer = Dynamic_Lightweight_Convolution(kernel_size = self.kernel_size,
                                                   h = self.h,
                                                   use_dynamic_weights = False)  
    elif self.memory_mechanism == 'persistant_highway_convolution':
      # based on preliminary experiments, the highway convolution and the persistant convolution both perform well
      # both operators, however, operate independentally and can be used concurrently
      # FLAGS: NOT IMPLEMENTED
      pass
    
  def assert_mechanism(self):
    # these are the only mechanisms currently offered
    if self.memory_mechanism == 'convolution':
      # the simple convolutional operator was used in arXiv:1803.01271
      # given the input shape [batch_size, sequence_size, hidden_size] and kernel size [kernel_size, hidden_size, hidden_size]
      # the kernel is convolved over the input to produce the output
      # if the operator is causal, the input is padded to the left, over the first axis, for (kernel_size - 1) time-steps with 0's
      # therefore, the output of time-step t is dependent upon the inputs x_t, x_{t - 1}, ..., x_{t - kernel_size}
      # the output at time-step t is not dependent on any time-step past t
      # if the operator is not causal, the input is convolved using the 'SAME' padding, allowing the output at time-step be dependent on
      # x_{t - kernel_size / 2}, ..., x_t, ..., x_{t + kernel_size / 2}
      # in arXiv:1803.01271, a layer of the model (TCN) is calculated as:
      # conv_1x1(inputs) + layer(layer(inputs)), where:
      # layer(x) = dropout(relu(norm(conv(x))))
      # given a kernel_size k and layers l, the output is capable of seeing (k-1)*(l-1) + k time-steps previous if causal
      ####################### Shakespeare Language Modelling ######################################
      # Train loss: 1.44, Train accuracy: 56%, Test loss: 1.89, Test accuracy: 48%, Time: 15.9924 #
      return True
    elif self.memory_mechanism == 'highway_convolution':
      # instead of a simple convolutional operator, the results are based on a highway network
      # instead of convolution, where the entire sequence is analyzed evenly, this allows the model to 'ignore' time-steps it learns to be useless
      # the output is defined as:
      # x * sigmoid(y) + inputs * (1 - sigmoid(y))
      # x, y = conv(inputs), conv(inputs)
      # the highway network was introduced in arXiv:1505.00387, and is considered capable of being trained for long-ranges
      ####################### Shakespeare Language Modelling ######################################
      # Train loss: 1.41, Train accuracy: 57%, Test loss: 1.84, Test accuracy: 48%, Time: 18.6687 #
      return True
    elif self.memory_mechanism == 'output_convolution':
      return True
    elif self.memory_mechanism == 'depthwise_convolution':
      # a traditional 1D convolutional operator requires the kernel to have shape [kernel_size, hidden_size, hidden_size]
      # the depthwise convolution kernel is of shape [kernel_size, hidden_size, 1]
      # this is the equivalent of, for each time-step, an element-wise multiplication rather then a matrix multiplication
      ####################### Shakespeare Language Modelling ######################################
      # Train loss: 1.5, Train accuracy: 53%, Test loss: 1.99, Test accuracy: 46%, Time: 14.9229 #  
      return True
    elif self.memory_mechanism == 'persistant_convolution':
      # for the traditional convolutional operator, described above, the input vector is padded with a 0's vector before analysis
      # the vector is of shape [1, kernel_size -1, hidden_size]. Regardless of the sequence of batch size, the size of the padding vector is static
      # this 0's vector, in this case, is replaced by a trainable vector of shape [1, kernel_size - 1, hidden_size]
      # this persistant vector can be trained in the most efficient manner for the model, in essence a permanent memory
      # the model is capable of creating a single vector to use across all layers, or a single vector to use for each layer
      ####################### Shakespeare Language Modelling ######################################
      # Train loss: 1.41, Train accuracy: 57%, Test loss: 2.02, Test accuracy: 47%, Time: 16.1305 #
      return True
    elif self.memory_mechanism == 'dilated_convolution':
      # the authors of arXiv:1803.01271 further introduced the dilated convolutional operator
      # the authors noted that a dilated convolution is capable to seeing further through time because of its dilation
      # that said, further authors have noted that systems, such as the active-memory system, is useful for modelling local relations
      # given that dilated memory extends the local field, the cost may outweight the cost
      # the larger receptive field can allow for greater long-range dependency, but cannot match the multihead-attention of the Transformer
      ####################### Shakespeare Language Modelling ######################################
      # Train loss: 1.46, Train accuracy: 56%, Test loss: 1.96, Test accuracy: 47%, Time: 16.7133 #
      return True
    elif self.memory_mechanism == 'dynamic_lightweight_conv':
      # the dynamic lightweight convolution was introduced in arXiv:1901.10430
      # a lightweight kernel, of size [kernel_size, h], is generated dynamically using a dense network
      # FLAGS: NEEDS FURTHER TESTING AND WRITE ABOUT RESULTS
      ####################### Shakespeare Language Modelling ######################################
      # Train loss: 1.64, Train accuracy: 51%, Test loss: 2.01, Test accuracy: 47%, Time: 56.3480 #
      return True
    elif self.memory_mechanism == 'lightweight_conv':
      # a depthwise convolution kernel is of shape [kernel_size, hidden_size]
      # a lightweight convolution kernel is of shape [kernel_size, h], which is then tiled hidden_size / h times
      # the lightweight kernel, post-tiling, is of shape [kernel_size, hidden_size], which is the same vector repeated hidden_size / h times
      # FLAGS: TEST THIS WHILE YOU'RE AT IT
      ####################### Shakespeare Language Modelling ######################################
      # FLAGS: this needs to be redone because I'm a fucking idiot
      return True
    elif 'depthwise_convolution' in self.memory_mechanism:
      # I've added highway and persistant convolutions based on experimental results
      return True
    return False
    
  def call(self, inputs):
    if self.causal and self.memory_mechanism != 'dilated_convolution':
      # if 'dilated_convolution', the keras layer automatically pads the inputs 
      if self.memory_mechanism != 'persistant_convolution':
        # if not 'persistant_convolution', the the input vector is padded with 0's
        inputs = tf.pad(inputs,
                        [[0, 0], 
                         [self.kernel_size - 1, 0], 
                         [0, 0]])
      else:
        # the persistant_vector is of shape [1, kernel_size - 1, hidden_size] initially
        # must be tiled to shape [batch_size, kernel_size - 1, hidden_size]
        # batch_size is dynamic in this case, so the tiling operation is conducted in a dynamic mmanner
        persistant_vector = tf.tile(self.persistant_vector,
                                    [tf.shape(inputs)[0], 1, 1])
        inputs = tf.concat([persistant_vector, inputs],
                           axis = 1)
      # for tensorflow convolution, with input [batch_size, sequence_size, hidden_size] and [kernel_size, hidden_size, hidden_size]
      # the output of the convolution if of shape [batch_size, sequence_size - kernel_size + 1, hidden_size] for padding = 'VALID'
      # therefore, given a zeros padding of shape [1, kernel_size - 1, hidden_size], the output maintains the same shape of the pre-padded input
      padding = 'VALID'
    else:
      # if the analysis is bidirectional, the padding = 'SAME'
      # this enables the model to analyze in a bidirectional manner, and maintain the smae shape
      # the input is padded on both the left and right side evenly
      padding = 'SAME'
    if self.memory_mechanism == 'convolution' or self.memory_mechanism == 'persistant_convolution':
      # inputs is already padded for both convolution and persistant
      # upon convolution, the output can be returned immediately
      # upon calculation, the output is sent through an activation function
      return self.activation(tf.nn.convolution(inputs,
                                               self.kernel,
                                               padding = padding))
    elif self.memory_mechanism == 'highway_convolution' or self.memory_mechanism == 'output_convolution':
      # the first output, with a linear activation function
      output = tf.nn.convolution(inputs,
                                 self.kernel,
                                 padding = padding)
      # the second output, with a sigmoid activation function
      sigmoid_output = sigmoid_cutoff(tf.nn.convolution(inputs,
                                                        self.sigmoid_kernel,
                                                        padding = padding))
      # the final output is calculated in a highway manner
      if self.causal:
        if self.memory_mechanism == 'highway_convolution':
          return tf.multiply(output, 
                             sigmoid_output) + tf.multiply(inputs[:,self.kernel_size - 1:],
                                                           1 - sigmoid_output)
        elif self.memory_mechanism == 'output_convolution':
          return inputs[:,self.kernel_size - 1:] + tf.multiply(sigmoid_output,
                                                               output)
      else:
        if self.memory_mechanism == 'highway_convolution':
          return tf.multiply(output, 
                             sigmoid_output) + tf.multiply(inputs,
                                                           1 - sigmoid_output)
        else:
          return inputs + tf.multiply(sigmoid_output,
                                      output)
    elif self.memory_mechanism == 'depthwise_convolution':
      # tensorflow does not offer a depthwise_1d under tf.nn
      # therefore input is expanded into shape [batch_size, 1, sequence_size, hidden_size]
      inputs = tf.expand_dims(inputs,
                              axis = 1)
      # kernel is of shape [1, kernel_size, hidden_size, output_multiplier]
      # across the first sequence axis (set to 1), forces this 2d convolution to act in an identical manner to a 1d convolution
      output = tf.nn.depthwise_conv2d(inputs,
                                      self.kernel,
                                      strides = [1, 1, 1, 1],
                                      padding = padding)
      # the output of the convolution is of shape [batch_size, 1, sequence_size, hidden_size]
      # the output is squeezed before being sent to the final activation function
      return self.activation(tf.squeeze(output,
                                        axis = 1))
    elif 'depthwise_convolution' in self.memory_mechanism:
      if 'persistant' in self.memory_mechanism:
        inputs = inputs[:,self.kernel_size - 1:]
        padding = 'VALID'
        persistant_vector = tf.tile(self.persistant_vector,
                                    [tf.shape(inputs)[0], 1, 1])
        inputs = tf.concat([persistant_vector, inputs],
                            axis = 1)
      inputs = tf.expand_dims(inputs,
                              axis = 1)  
      output = tf.nn.depthwise_conv2d(inputs,
                                      self.kernel,
                                      strides = [1, 1, 1, 1],
                                      padding = padding)
      if 'highway' in self.memory_mechanism:
        sigmoid_output = sigmoid_cutoff(tf.nn.depthwise_conv2d(inputs,
                                                               self.sigmoid_kernel,
                                                               strides = [1, 1, 1, 1],
                                                               padding = padding))
        inputs = inputs[:,:,self.kernel_size - 1:]
        return tf.squeeze(output * sigmoid_output + inputs * (1 - sigmoid_output),
                          axis = 1)
      return tf.squeeze(output,
                        axis = 1)
    elif self.memory_mechanism == 'dilated_convolution':
      # for each 'dilated_convolution', 'dynamic_lightweight_conv' and 'lightweight_conv', a predefined keras layer is used
      # an activation function is applied to each afterwards
      return self.activation(self.layer(inputs))
    elif self.memory_mechanism == 'dynamic_lightweight_conv':
      return self.activation(self.layer(inputs))
    elif self.memory_mechanism == 'lightweight_conv':
      return self.activation(self.layer(inputs))

class Adaptive_Span(tf.keras.layers.Layer):
  # adaptive attention span, as suggested in arXiv:1905.07799
  # FLAGS: link up parameters to transformer.py
  def __init__(self, dynamic_attention_span = False,
               init_val = 0.0,
               ramp_size = 32,
               max_span = None,
               **kwargs):
    super(Adaptive_Span,
          self).__init__(**kwargs)
    self.dynamic_attention_span = dynamic_attention_span # True
    # whether to use dynamic attention span or static attention span
    # for the time being, this can only be implemented with self-attention
    self.init_val = init_val
    # the initial value of the weight value
    self.ramp_size = ramp_size
    # the R hyperparameter that controls its softness
    self.max_span = max_span
    # the maximum span of the attention head
    # if set to None, then the maximum span is the input sequence size
    
  def build(self, input_shape):
    self.num_heads = input_shape.as_list()[1]
    if self.dynamic_attention_span:
      self.current_val = tf.keras.layers.Dense(1,
                                               activation = sigmoid_cutoff,
                                               name = 'current_val')
    else:
      # if the static attention span is used, then a trainable value is used
      # the value is of size [1, num_heads, 1, 1]
      # inputs is of size [batch_size, num_heads, seq_len, seq_len]
      self.current_val = self.add_weight('current_val',
                                         shape = [1, self.num_heads, 1, 1],
                                         initializer = tf.zeros_initializer(),
                                         dtype = tf.float32) + self.init_val
      # the parameter is specified to be [0, span]
      # the parameter is multipled by span later, and is clipped later as well
      
  def call(self, inputs,
           attention_output = None):
    # attention_output is the tensor of shape [batch_size, num_heads, seq_len, head_size]
    # the final layer must be static to be sent through a dense layer to collect the dynamic current_val
    if self.dynamic_attention_span:
      current_val = self.current_val(attention_output)
      # current_val is of shape [batch_size, num_heads, sequence_size, 1]
    else:
      current_val = self.current_val
    attn_span = tf.cast(tf.shape(inputs)[-2],
                        dtype = tf.float32)
    # e.g. given attn_span = 10
    # mask_template = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    mask_template = tf.range(0,
                             attn_span,
                             delta = 1,
                             dtype = tf.float32)
    if self.max_span is not None:
      # if self.max_span is not None, then the attn_span is clipped below the max_span
      attn_span = tf.minimum(attn_span,
                             self.max_span)
      mask_template = tf.minimum(mask_template,
                                 self.max_span)
    mask_template = tf.reshape(mask_template,
                               [1, 1, 1, -1])
    # mask template is of shape [1, 1, 1, seq_len]
    # current_val is of shape shape [1, num_heads, 1, 1] or above
    mask = mask_template + current_val * attn_span
    mask /= (self.ramp_size + 1)
    mask = tf.clip_by_value(mask,
                            0,
                            1)
    memory_mask = tf.ones([1, self.num_heads, 1, tf.shape(inputs)[-1] - tf.shape(inputs)[-2]],
                          dtype = tf.float32)
    if not self.dynamic_attention_span:
      mask = tf.concat([memory_mask, mask],
                       axis = -1)
    # the mask is divided by a ramp_size and clipped between 0 and 1
    # given example above, and ramp_size of 6
    # mask = [0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1, 7/6, 4/3, 3/2]
    inputs = mask*inputs
    # mask e [1, num_heads, 1, seq_len]
    # inputs e [batch_size, num_heads, seq_len, seq_len]
    inputs /= (tf.reduce_sum(inputs,
                             axis = -1,
                             keepdims = True) + 1e-8)
    l0 = tf.reduce_mean(current_val) * attn_span
    self.add_loss(l0 * self.num_heads ** -1)
    # the loss is found using model.losses and used as an auxiallary loss
    return inputs
    
class Bias_Add(tf.keras.layers.Layer):
  # a keras layer that simply adds a bias vector, similar to a ffd layer without the weight W
  # with use of this as a keras layer, the implementation is much easier
  def __init__(self, hidden_size,
               initializer = tf.zeros_initializer(),
               **kwargs):
    # hidden_size - the hidden_size of the input vector
    # initializer - the initialization mechanism of the bias vector
    super(Bias_Add,
          self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.initializer = initializer
    
  def build(self, input_shape):
    self.bias = self.add_weight('bias',
                                shape = [self.hidden_size],
                                dtype = tf.float32,
                                initializer = self.initializer)
                                
  def call(self, inputs):
    return inputs + self.bias

class Compression(tf.keras.layers.Layer):
  # returns a convolutional kernel for compression, as stated in arXiv:1911.05507
  # FLAGS: Highway? Persistant?
  def __init__(self, compression_rate,
               **kwargs):
    super(Compression,
          self).__init__(**kwargs)
    self.compression_rate = compression_rate
    
  def build(self, input_shape):
    hidden_size = input_shape.as_list()[-1]
    self.compression_kernel = self.add_weight('compression-kernel',
                                              shape = [1, self.compression_rate, hidden_size, hidden_size],
                                              dtype = tf.float32)
                                              
  def call(self, inputs):
    # the vector must be sent through the the convolution here
    # if it is not, then the trainable variable does not appear in the keras model
    compressed_vector = tf.nn.convolution(inputs,
                                          self.compression_kernel,
                                          strides = [1, 1, self.compression_rate, 1],
                                          padding = 'VALID')
    return compressed_vector

class Dynamic_Lightweight_Convolution(tf.keras.layers.Layer):
  # a dynamic lightweight convolution layer, as defined in arXiv:1901.10430
  def __init__(self, kernel_size,
               h,
               use_dynamic_weights = False,
               **kwargs):
    # kernel_size - the size of the kernel_size
    # h - the parameter of the tied weights
    # use_dynamic_weights - whether to use dynamic_weights, or simple lightweight_conv
    super(Dynamic_Lightweight_Convolution,
          self).__init__(**kwargs)
    self.kernel_size = kernel_size
    self.h = h
    self.use_dynamic_weights = use_dynamic_weights
    
  def build(self, input_shape):
    self.hidden_size = input_shape[-1]
    if self.use_dynamic_weights:
      # if use_dynamic_weights, then the weights are generated by an ffd
      # the output size of ffd is kernel_size * h, so the output can be reshape to [batch_size, sequence_size, kernel_size, h]
      # this is tiled to the shape [batch_size, sequence_size, kernel_size, hidden_size]
      # if kernel_size = 7, h = 16, hidden_size = 256, then to calculate the lightweight-kernel requires a dense with output 112
      # else, a depthwise kernel would require an output size of 1792
      self.dynamic_weights = tf.keras.layers.Dense(self.kernel_size * self.h)
    else:
      # the lightweight kernel is created statically to be of shape [kernel_size, h]
      # this function is identical to a depthwise convolution, except that the kernel, while of identical shape, has less parameters
      self.lightweight_kernel = self.add_weight('lightweight_kernel',
                                                shape = [self.kernel_size, self.h],
                                                dtype = tf.float32)
      # as arXiv:1901.10430 states, the kernel is subject to a softmax over the kernel axis
      # FLAGS: is softmax necessary?
      # FLAGS: is softmax over the axis = 0 correct, or axis = 1
      self.lightweight_kernel = tf.nn.softmax(self.lightweight_kernel,
                                              axis = 0)
      # there are kernel_size * h trainable parameters, but they are tiled to represent kernel_size * hidden_size parameters
      self.lightweight_kernel = tf.tile(self.lightweight_kernel,
                                        [1, self.hidden_size // self.h])
      self.lightweight_kernel = tf.reshape(self.lightweight_kernel,
                                           [1, self.kernel_size, self.hidden_size, 1])
      
  def call(self, inputs):
    if self.use_dynamic_weights:
      sequence_size = tf.shape(inputs)[1] - self.kernel_size + 1
      # the lightweight_kernel is generated dynamically, or shape [batch_size, sequence_size, kernel_size * h]
      lightweight_kernel = self.dynamic_weights(inputs[:,self.kernel_size - 1:])
      lightweight_kernel = tf.reshape(lightweight_kernel,
                                      [tf.shape(inputs)[0], tf.shape(inputs)[1] - self.kernel_size + 1, self.kernel_size, self.h])
      lightweight_kernel = tf.nn.softmax(lightweight_kernel,
                                         axis = -2)
      # the lightweight_kernel is tiled to be of shape [batch_size, sequence_size, kernel_size, hidden_size]
      # the kernel is generated dynamically
      lightweight_kernel = tf.tile(lightweight_kernel,
                                   [1, 1, 1, self.hidden_size // self.h])
      # tensorflow does not offer a dynamic convolution operation, to the best of my knowledge, therefore I have had to write it personally
      # the operation is conducted using a while_loop
      output = []
      i = tf.constant(0)
      new_state = tf.zeros_like(inputs[:,0:1])
      def cond(i, state, new_state):
        # while the iteration is less then the sequence size, the loop continues
        return tf.not_equal(i, sequence_size)
      def body(i, state, new_state):
        # given an iteration i, such that 0=< i < sequence_size
        # the current state is calculated as inputs[:,i:i+kernel_size], [batch_size, kernel_size, hidden_size]
        # the current kernel is calculcated as lightweight_kernel[:,i], [batch_size, kernel_size, hidden_size]
        # the output is multiplied and avearaged over the time axis
        time_state = state[:,i:i + self.kernel_size]
        new_kernel = lightweight_kernel[:,i]
        time_state = tf.reduce_mean(new_kernel * time_state,
                                    axis = 1,
                                    keepdims = True)
        # the state, for computational reasons, starts as a zeros vector of shape [batch_size, 1, hidden_size]
        # the state is concatenated along the time axis to the right
        new_state = tf.concat([new_state, time_state],
                              axis = 1)
        return i + 1, state, new_state
      
      _, _, output = tf.while_loop(cond,
                                   body,
                                   [i, inputs, new_state],
                                   shape_invariants = [i.get_shape(), inputs.get_shape(), tf.TensorShape([None, None, self.hidden_size])])
      # becauset the output starts as a zeros vector of shape [batch_size, 1, hidden_size], the output is [batch_size, sequence_size + 1, hidden_size]
      # the first time-step is 0's. It can be deleted
      return output[:,1:]
    else:
      # because this is nn.depthwise_conv2d, the first sequence_size axis must be expanded
      inputs = tf.expand_dims(inputs,
                              axis = 1)
      output = tf.nn.depthwise_conv2d(inputs,
                                      self.lightweight_kernel,
                                      strides = [1, 1, 1, 1],
                                      padding = 'VALID')
      return tf.squeeze(output, 
                        axis = 1)
                        
class GLU(tf.keras.layers.Layer):
  def __init__(self, kernel_size,
               output_size = None,
               use_highway = False,
               unidirectional = True,
               **kwargs):
    # a gated linear unit, as specified in arXiv:1612.08083
    # if use_highway = True, it becomes a highway unit, as specified in arXiv:1505.00387
    # GLU is specified as:
    # conv(inputs) * sigmoid(conv(inputs))
    super(GLU,
          self).__init__(**kwargs)
    self.kernel_size = kernel_size
    self.output_size = output_size
    self.use_highway = use_highway
    self.unidirectional = unidirectional
    # kernel_size - the kernel_size of the convolutional operator
    # if set to 1, then the convolution operators become a dense operator
    # output_size - the output size. If output_size is None, then it is set as input_size
    # use_hightway - whether to use a highway network or simply a GLU network
    # unidirectional - if unidirectional, the inputs are padded to the left
    
  def build(self, input_shape):
    input_size = input_shape.as_list()[-1]
    if self.output_size is None:
      self.output_size = input_size
    self.kernel = self.add_weight('kernel',
                                  shape = [self.kernel_size, input_size, self.output_size],
                                  dtype = tf.float32)
    self.bias = self.add_weight('bias', 
                                shape = [self.output_size],
                                dtype = tf.float32)
    self.sigmoid_kernel = self.add_weight('sigmoid-kernel',
                                          shape = [self.kernel_size, input_size, self.output_size],
                                          dtype = tf.float32)
    self.sigmoid_bias = self.add_weight('sigmoid-bias',
                                        shape = [self.output_size],
                                        dtype = tf.float32)
  
  def call(self, inputs):
    if self.unidirectional:
      inputs = tf.pad(inputs,
                      [[0, 0],
                       [self.kernel_size - 1, 0],
                       [0, 0]])
    output = tf.nn.convolution(inputs,
                               self.kernel,
                               padding = 'VALID' if self.unidirectional else 'SAME') + self.bias
    sigmoid_output = sigmoid_cutoff(tf.nn.convolution(inputs,
                                                      self.sigmoid_kernel,
                                                      padding = 'VALID' if self.unidirectional else 'SAME') + self.sigmoid_bias)
    if self.use_highway:
      return output * sigmoid_output + inputs * (1 - sigmoid_output)
    else:
      return output * sigmoid_output
    
class LocalRNN(tf.keras.layers.Layer):
  # the LocalRNN is defined in arXiv:1907.05572
  # RNN's are capable of analyzing sequences in a recurrent manner, being able to deduce recurrent patterns
  # at the cost of an inability to analyze long-range dependencies effectively and at substantial computational time
  # Transformers are the opposite
  # Transformers can analyze (theoretically infinite) long-range dependencies at a reasonable computational time
  # but cannot deduce recurrent patterns
  # the LocalRNN was introduced as a module that attempts to combine the two
  # the authors of arXiv:1907.05572 designed the following model:
  # FFD(Attention(LocalRNN(x))) for each layer
  # by using these two functions, one after another, the self-attention mechanism is theoretically capable of performing its normal functions
  # and deduce recurrent connections
  # the LocalRNN is not a traditional RNN
  # instead of analyzing an entire sequence, y_t = RNN(x_t, h_t)
  # the output at time t is instead calculcated as y_t = RNN(x_t to x_{t-w})
  # instead of analyzing the entire sequence (require t time-steps), the sequence is analyzed in window sizes of w (requiring w time-steps)
  # this has the advantage of severely limiting the computational time
  # however, the output at time-step t are dependent on only the previous w time-steps
  # this inhibits the RNN's ability to analyze long-rage dependencies, which it cannot easily do and the multihead-attention can perform better
  def __init__(self, window_size,
               use_peephole = False,
               use_vanilla_rnn = True,
               **kwargs):
    super(LocalRNN,
          self).__init__(**kwargs)
    self.window_size = window_size
    self.use_peephole = use_peephole
    self.use_vanilla_rnn = use_vanilla_rnn
    # window_size - the output at time-step t is dependent on the inputs x_t - x_{t - w}
    # use_peeophole - if LSTM is used, then a peeophole LSTM is used if True
    # use_vanilla_rnn - whether to use a traditional vanilla RNN or an LSTM
    # FLAGS: build and test GRU and MGU (arXiv:1701.03452)
    
  def build(self, input_shape):
    self.hidden_size = input_shape[-1]
    if self.use_vanilla_rnn:
      self.x_layer = tf.keras.layers.Dense(self.hidden_size,
                                           activation = tf.identity,
                                           use_bias = False)
      self.y_layer = tf.keras.layers.Dense(self.hidden_size * 2,
                                           activation = tf.identity,
                                           use_bias = True)
    else:
      self.x_layer = tf.keras.layers.Dense(self.hidden_size * 4,
                                           activation = tf.identity,
                                           use_bias = False)
      self.y_layer = tf.keras.layers.Dense(self.hidden_size * 4,
                                           activation = tf.identity,
                                           use_bias = True)
                                         
  def call(self, inputs):
    sequence_size = tf.shape(inputs)[1]
    # the initial state if a zeros state of shape [batch_size, sequence_size, hidden_size]
    state = tf.zeros_like(inputs)
    
    # context is similar to state, but only used for LSTM
    context = tf.zeros_like(inputs)
    # similar to the Active-Memory the inputs must be padded for window_size - 1
    # therefore, the output y_1 is dependent only on x_1
    inputs = tf.pad(inputs,
                    [[0, 0], [self.window_size - 1, 0], [0, 0]])
    # the inputs are not dependent on time, therefore the dense of inputs should be calculated beforehand to minimize computational time
    inputs = self.x_layer(inputs)
    # because window_size is static, the calculation can be calculcated statically
    # the output, y_t, is only dependent on the previous w time-steps, not the previous t
    # therefore, amount of previous time-steps to look back upon is static and window_size
    for i in range(self.window_size):
      state, context = self.time_step(inputs[:,i:sequence_size + i],
                                      state,
                                      context)
    return state
    
  def time_step(self, inputs,
                state,
                old_context):
    batch_size, sequence_size = tf.shape(inputs)[0], tf.shape(inputs)[1]
    if self.use_vanilla_rnn:
      # for vanilla_rnn:
      # new_state = dense(state)
      # gate = sigmoid(dense(state))
      # output = ReLU(gate * (inputs + new_state) + (1 - gate) * state)
      # FLAGS: gate = sigmoid(dense(state + inputs)) ???
      new_state = self.y_layer(state)
      new_state = tf.reshape(new_state,
                             [batch_size, sequence_size, 2, self.hidden_size])
      gate = sigmoid_cutoff(new_state[:,:,1])
      new_state = new_state[:,:,0]
      # FLAGS: could this benefit from losing the ReLU. Also relu/gelu
      return tf.nn.relu(tf.multiply(gate,
                                    inputs + new_state) + tf.multiply(1 - gate,
                                                                      state)), None
    else:
      # for LSTM
      # f = dense(inputs) + dense(state)
      # i = dense(inputs) + dense(state)
      # o = dense(inputs) + dense(state)
      # new_context = f * old_context + i * tanh(dense(inputs) + dense(state))
      # output = o * tanh(new_context)
      # OR, if peephole
      # output = o * new_context
      batch_size, sequence_size = tf.shape(inputs)[0], tf.shape(inputs)[1]
      inputs = tf.reshape(inputs,
                          [batch_size, sequence_size, 4, self.hidden_size])
      state = self.y_layer(state)
      state = tf.reshape(state,
                         [batch_size, sequence_size, 4, self.hidden_size])
      f_gate = sigmoid_cutoff(inputs[:,:,0] + state[:,:,0])
      i_gate = sigmoid_cutoff(inputs[:,:,1] + state[:,:,1])
      o_gate = sigmoid_cutoff(inputs[:,:,2] + state[:,:,2])
      new_context = f_gate * old_context + i_gate * tf.nn.tanh(inputs[:,:,3] + state[:,:,3])
      if self.use_peephole:
        return o_gate * new_context, new_context
      else:
        return o_gate * tf.nn.tanh(new_context), new_context

class MoS(tf.keras.layers.Layer):
  # the authors of arXiv:1711.03953 note that traditional output functions for RNNs are not capable of effectively handling large vocabulary outputs
  # quote: "standard Softmax-based lan-guage models with distributed (output) word embeddings do not have enough capacity to modelnatural language"
  # the solution was the Mixture-of-Softmax, which is capable of overcoming these problems
  def __init__(self, vocab_size,
               n_experts,
               **kwargs):
    super(MoS,
          self).__init__(**kwargs)
    self.vocab_size = vocab_size
    # the vocab_size of the output
    self.n_experts = n_experts
    # the number of experts used for analysis
    
  def build(self, input_shape):
    self.hidden_size = input_shape.as_list()[-1]
    self.latent_dense = tf.keras.layers.Dense(self.n_experts * self.hidden_size,
                                              activation = tf.nn.tanh)
    # if tanh an effective choice of activation function?
    # Shakespeare task, batch_size = 128
    # GeLU
    # Epoch 10, Loss 1.47, Accuracy 0.55
    # Test Loss 2.02, Accuracy 0.47
    # Tanh
    # Epoch 10, Loss 1.39, Accuracy 0.58
    # Test Loss 1.94, Accuracy 0.48
    # tanh out-performs GeLU
    # further: batch_size = 1024
    # Epoch 10, Loss 1.25, Accuracy 0.61
    # Test Loss 1.52, Accuracy 0.54
    self.logit_dense = tf.keras.layers.Dense(self.vocab_size)
    self.prior_dense = tf.keras.layers.Dense(self.n_experts)
    
  def call(self, inputs):
    # FLAGS: comment this function through
    batch_size = tf.shape(inputs)[0]
    latent_vector = self.latent_dense(inputs)
    latent_vector = tf.reshape(latent_vector,
                               [batch_size, -1, self.n_experts, self.hidden_size])
    logit_vector = self.logit_dense(latent_vector)
    prior_logit_vector = self.prior_dense(inputs)
    prior_logit_vector = tf.nn.softmax(prior_logit_vector,
                                       axis = -1)
    prob_vector = tf.nn.softmax(logit_vector,
                                axis = -1)
    prior_logit_vector = tf.expand_dims(prior_logit_vector,
                                        axis = 3) # 2
    prior_logit_vector = tf.tile(prior_logit_vector,
                                 [1, 1, 1, self.vocab_size])
    prob_vector *= prior_logit_vector
    prob_vector = tf.reduce_sum(prob_vector,
                                axis = 2)
    return prob_vector
    
class Multihead_Attention(tf.keras.layers.Layer):
  # the multihead attention mechanism, as specified by arXiv:1706.03762
  def __init__(self, head_size,
               num_heads,
               share_qk = False,
               unidirectional = True,
               adaptive_span = False,
               dynamic_adaptive_span = False,
               **kwargs):
    super(Multihead_Attention,
          self).__init__(**kwargs)
    self.head_size = head_size
    self.num_heads = num_heads
    self.share_qk = share_qk
    self.unidirectional = unidirectional
    self.adaptive_span = adaptive_span
    self.dynamic_adaptive_span = dynamic_adaptive_span
    
  def build(self, input_shape):
    self.hidden_size = input_shape.as_list()[-1]
    self.query_dense = tf.keras.layers.Dense(self.num_heads * self.head_size,
                                             use_bias = False)
    # i've tried to deparameterize the self-attention mechanism by forcing the query, key and value through the same dense layer
    # it was suboptimal
    # authors of arXiv:2001.04451 suggested that the query and key could be the same dense layer
    if not self.share_qk:
      self.key_dense = tf.keras.layers.Dense(self.num_heads * self.head_size,
                                             use_bias = False)
    self.value_dense = tf.keras.layers.Dense(self.num_heads * self.head_size,
                                             use_bias = False)
    self.output_dense = tf.keras.layers.Dense(self.hidden_size,
                                              use_bias = False)
    if self.adaptive_span:
      self.adaptive_span_layer = Adaptive_Span(dynamic_attention_span = self.dynamic_adaptive_span)
                                              
  def call(self, query,
           training,
           dropout_rate,
           key_value = None,
           mask = None):
    if key_value == None:
      key_value = query
      # if key_value = query, then this is self-attention, the vector is analyzed across itself
      # however, if key_value is not query, then the input is analyzed against another sequence
      # e.g. machine translation, where the output is dependent on the analysis of the input in another language
    query = self.query_dense(query)
    # query e [batch_size, sequence_size, num_heads * head_size]
    if self.share_qk:
      key = self.query_dense(key_value)
    else:
      key = self.key_dense(key_value)
    # key e [batch_size, sequence_size, num_heads * head_size]
    value = self.value_dense(key_value)
    # value e [batch_size, sequence_size, num_heads * head_size]
    query, key, value = (self.split_heads(vector) for vector in [query, key, value])
    # query, key, value e [batch_size, num_heads, sequence_size, head_size]
    
    weights = tf.matmul(query,
                        key,
                        transpose_b = True)
    # weights e [batch_size, num_heads, sequence_size, sequence_size]
    # this is a theoretically endless long-range dependency, for each sequence in the -2 axis, each possible sequence_size can be seen
    weights /= self.head_size
    # the authors of arXiv:1706.03762 note that the weights must be scaled
    if self.unidirectional:
      # if bidirectional, then the model should be able to connect any two sequences
      bias = self.build_bias(tf.shape(weights)[-1])
      weights += bias
    # I replaced this with a sigmoid rather then a softmax, so the model could focus on many different time-steps at once
    # conceptually it doesn't work, but you never know for sure until you test
    # I tested it, and it didn't work
    if mask is not None:
      mask = tf.expand_dims(mask,
                            axis = 1)
      mask = tf.expand_dims(mask,
                            axis = 1)
      mask *= -1e9
      weights += mask
    weights = tf.nn.softmax(weights,
                            axis = -1)
    if self.adaptive_span:
      weights = self.adaptive_span_layer(weights,
                                         attention_output = query)
    
    # weights are subject to a softmax across the final axis
    # this, is essence, forces the sequence to only look at one other sequence per head
    # therefore, given 8 heads, the sequence is forced to focus on 8 other sequences
    # on one hand, this inability to focus on multiple is a theoretical problem (see Active_Memory)
    # on the other hand, experimental results suggest that most heads can be pruned
    weights = tf.keras.layers.Dropout(dropout_rate)(weights,
                                                    training = training)
    output = tf.matmul(weights,
                       value)
    # output e [batch_size, num_heads, sequence_size, head_size]
    output = tf.transpose(output,
                          [0, 2, 1, 3])
    # output e [batch_size, sequence_size, num_heads, head_size]
    output = tf.reshape(output,
                        [tf.shape(output)[0],
                         tf.shape(output)[1],
                         self.num_heads * self.head_size])
    # output e [batch_size, sequence_size, num_heads * head_size]
    return self.output_dense(output)
    
  def build_bias(self, sequence_size):
    # examples have sequence_size of 5
    attn_mask = tf.ones([sequence_size, sequence_size])
    mask_u = tf.linalg.band_part(attn_mask,
                                 0,
                                 -1)
    # mask_u = [[1, 1, 1, 1, 1],
    #           [0, 1, 1, 1, 1],
    #           [0, 0, 1, 1, 1],
    #           [0, 0, 0, 1, 1],
    #           [0, 0, 0, 0, 1]]
    mask_dia = tf.linalg.band_part(attn_mask,
                                   0,
                                   0)
    # mask_dia = [[1, 0, 0, 0, 0],
    #             [0, 1, 0, 0, 0],
    #             [0, 0, 1, 0, 0],
    #             [0, 0, 0, 1, 0],
    #             [0, 0, 0, 0, 1]]
    
    # local connections only
    # add to already existing
    # local_mask = tf.linalg.band_part(attn_mask, -1, 0)
    # local_dia = tf.linalg.band_part(attn_mask, kernel_size, 0)
    # local_mask - local_dia
    # mask_u += local_mask
    # FLAGS: give this a try
    # I think https://arxiv.org/pdf/2002.06170.pdf did this
    
    bias = (mask_u - mask_dia) * -1e9
    # bias = [[0, -1e9, -1e9, -1e9, -1e9],
    #         [0, 0, -1e9, -1e9, -1e9],
    #         [0, 0, 0, -1e9, -1e9],
    #         [0, 0, 0, 0, -1e9],
    #         [0, 0, 0, 0, 0]]
    # bias[0] = [0, -1e9, ...]
    # the nth bias has n 0's, followed only by 1e-9
    # by setting a number to 1e-9, the softmax value goes to 0
    # this cuts any possible unwanted connections
    return tf.reshape(bias,
                      [1, 1, sequence_size, sequence_size])
    
  def split_heads(self, tensor):
    # tensor e [batch_size, sequence_size, num_heads, head_size]
    # output e [batch_size, num_heads, sequence_size, head_size]
    # this is because the sequence_size axis is the crux of the analysis
    # moving the sequence_size around allows for an ease of functionality
    shape = [tf.shape(tensor)[0],
             tf.shape(tensor)[1],
             self.num_heads,
             self.head_size]
    tensor = tf.reshape(tensor,
                        shape = shape)
    return tf.transpose(tensor,
                        [0, 2, 1, 3])
    
class Persistant_Vector(tf.keras.layers.Layer):
  # creates a persistant vector that can be used globally by the Active-Memory system
  def __init__(self, kernel_size,
               **kwargs):
    super(Persistant_Vector,
          self).__init__(**kwargs)
    self.kernel_size = kernel_size
    
  def build(self, input_shape):
    self.hidden_size = input_shape.as_list()[-1]
    self.persistant_vector = self.add_weight('persistant-vector',
                                             shape = [1, self.kernel_size - 1, self.hidden_size],
                                             dtype = tf.float32)
    
  def call(self, inputs):
    return self.persistant_vector
    
class Positional_Embedding(tf.keras.layers.Layer):
  # a positional embedding adds a trainable vector across the input tensor, with each time-step have a different trainable vector
  # this explicitly allows the model to learn how to interpret the position of each token based on this vector
  # the drawback is the fact that the maximum sequence size must be explicitly stated when the model is built
  def __init__(self, max_length,
               **kwargs):
    super(Positional_Embedding,
          self).__init__(**kwargs)
    self.max_length = max_length
    
  def build(self, input_shape):
    self.hidden_size = input_shape.as_list()[-1]
    self.kernel = self.add_weight(name = 'positional-kernel',
                                  shape = [self.max_length, self.hidden_size],
                                  dtype = tf.float32)
    
  def call(self, inputs):
    sequence_size = tf.shape(inputs)[1]
    sliced = tf.expand_dims(self.kernel,
                            axis = 0)[:,:sequence_size,:]
    # self.kernel [max_length, hidden_size]
    # sliced [1, sequence_size, hidden_size]
    sliced = tf.tile(sliced,
                     [tf.shape(inputs)[0], 1, 1])
    # sliced [batch_size, sequence_size, hidden_size]
    return sliced + inputs

class Power_Normalization(tf.keras.layers.Layer):
  # power normalization is defined in arXiv:2003.07845
  def __init__(self, momentum = 0.99,
               **kwargs):
    super(Power_Normalization,
          self).__init__(**kwargs)
    self.momentum = momentum
          
  def build(self, input_shape):
    hidden_size = input_shape.as_list()[-1]
    self.sigma = self.add_weight(name = 'sigma',
                                 shape = (),
                                 dtype = tf.float32,
                                 initializer = tf.zeros_initializer(),
                                 trainable = True)
                                
  def call(self, inputs):
    return inputs * self.sigma

class Relative_Bias(tf.keras.layers.Layer):
  # if the relative bias vectors are global, then they need to be called and inputted into each relative-attention mechanism
  # this layer creates and returns the layer
  def __init__(self, num_heads,
               head_size,
               **kwargs):
    super(Relative_Bias,
          self).__init__(**kwargs)
    self.num_heads = num_heads
    self.head_size = head_size
    
  def build(self, input_shape):
    self.r_w_bias = self.add_weight('r_w_bias',
                                    shape = [1, self.num_heads, 1, self.head_size],
                                    dtype = tf.float32)
    self.r_r_bias = self.add_weight('r_r_bias',
                                    shape = [1, self.num_heads, 1, self.head_size],
                                    dtype = tf.float32)
                                    
  def call(self, inputs):
    return self.r_w_bias, self.r_r_bias

class Relative_Multihead_Attention(Multihead_Attention):
  def __init__(self, tie_weights = True,
               layer_norm = True,
               **kwargs):
    self.tie_weights = tie_weights
    self.layer_norm = layer_norm
    # this is a superclass of keras layers, but as a Multihead_Attention superclass
    # this allows a lot of the functions from the Multihead_Attention to be recycled
    super(Relative_Multihead_Attention,
          self).__init__(**kwargs)
          
  def attention_reconstruction(self, query,
                               key_value):
    # the pseudo-attention mechanism here does not require extensive details
    # it's a basic skeleton of relative-attention for the use of compression
    # no actual output goes through here
    # because the compressed vectors have an extended first-axis, the axis must be squeezed
    key_value = tf.squeeze(key_value,
                           axis = 1) 
    # FLAGS: the compression algorithm as stated in arXiv:1911.05507, analyzes the current output by two key/values:
    # - the uncompressed vector
    # - its compressed equivalent
    # and takes the loss of the two outputs to train
    # how sure am I it shouldn't be query equal to the uncompressed vector ?
    query = self.query_dense(query)
    if self.share_qk:
      key = self.query_dense(key_value)
    else:
      key = self.key_dense(key_value)
    value = self.value_dense(key_value)
    query, key, value = (self.split_heads(vector) for vector in [query, key, value])
    weights = tf.matmul(query,
                        key,
                        transpose_b = True)
    output = tf.matmul(tf.nn.softmax(weights,
                                     axis = -1),
                       value)
    output = tf.transpose(output,
                          [0, 2, 1, 3])
    output = tf.reshape(output,
                        [tf.shape(output)[0],
                         tf.shape(output)[1],
                         self.num_heads * self.head_size])
    output = self.output_dense(output)
    return output
  
  def build(self, input_shape):
    self.hidden_size = input_shape.as_list()[-1]
    self.query_dense = tf.keras.layers.Dense(self.num_heads * self.head_size,
                                             use_bias = False)
    if not self.share_qk:
      self.key_dense = tf.keras.layers.Dense(self.num_heads * self.head_size,
                                             use_bias = False)
    self.value_dense = tf.keras.layers.Dense(self.num_heads * self.head_size,
                                             use_bias = False)
    self.output_dense = tf.keras.layers.Dense(self.hidden_size,
                                              use_bias = False)
    self.relative_dense = tf.keras.layers.Dense(self.num_heads * self.head_size,
                                                use_bias = False)
    if self.layer_norm:
      self.memory_layer_norm = tf.keras.layers.LayerNormalization()
      self.query_layer_norm = tf.keras.layers.LayerNormalization()
    if self.adaptive_span:
      self.adaptive_span_layer = Adaptive_Span()
    if not self.tie_weights:
      self.r_w_bias = self.add_weight('r_w_bias',
                                      shape = [1, self.num_heads, 1, self.head_size],
                                      dtype = tf.float32)
      self.r_r_bias = self.add_weight('r_r_bias',
                                      shape = [1, self.num_heads, 1, self.head_size],
                                      dtype = tf.float32)
                                                
  def call(self, query,
           training,
           dropout_rate,
           r_w_bias = None,
           r_r_bias = None,
           key_value = None,
           memory = None,
           mask = None):
    # if there are inputs that does not need attention (such as padding), then mask can remove these inputs from consideration
    if r_w_bias is not None and r_r_bias is not None:
      self.r_w_bias = r_w_bias
      self.r_r_bias = r_r_bias
      # if r_w_bias and r_r_bias vectors are global, then the inputs here are not None
      # else, the inputs here are the None and the local biases were created in self.build()
    if key_value is None:
      key_value = query
    if memory is not None:
      self.memory_size = tf.shape(memory)[1]
      key_value = tf.concat([memory, key_value],
                            axis = 1)
      # the memory is concatenated to the left of key_value
      # afterwards, the memory is sent through a layer_norm
      if self.layer_norm:
        key_value = self.memory_layer_norm(key_value)
    else:
      self.memory_size = 0
      if self.layer_norm:
        key_value = self.query_layer_norm(key_value)
    # query is sent through a seperate layer norm because, if memory is used, the memory is concatenated and changes the vector
    # FLAGS: is this a good idea?
    if self.layer_norm:
      query = self.query_layer_norm(query)
    
    # positional embedding is similar to the Timing_Signal, put used differently    
    self.input_size = tf.shape(query)[1]
    pos_seq = tf.range(self.input_size + self.memory_size - 1,
                       -1,
                       -1.0)
    # e.g. [100.0, 99.0, 98.0, ..., 2.0, 1.0, 0.0]
    # pos_seq e [input_size + memory_size]
    inv_freq = 1 / (10000 ** (tf.range(0, self.hidden_size, 2.0) / self.hidden_size))
    # e.g. [1.0, 0.749, 0.562, 0.412, ...]
    # inv_freq e [hidden_size // 2]
    sinusoid_inp = tf.einsum('i,j->ij',
                             pos_seq,
                             inv_freq)
    # sinusoid_inp e [sequence_size, hidden_size // 2]
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)],
                        axis = -1)
    # pos_emb e [sequence_size, hidden_size]
    pos_emb = tf.tile(pos_emb[None,:,:],
                      [tf.shape(query)[0], 1, 1])
    # pos_emb e [batch_size, sequence_size, hidden_size]
    
    query = self.query_dense(query)
    if self.share_qk:
      key = self.query_dense(key_value)
    else:
      key = self.key_dense(key_value)
    value = self.value_dense(key_value)
    # unlike traditional Timing_Signal, the pos_emb is not simply added but specifically used in the attention architecture
    relative_bias = self.relative_dense(pos_emb)
    query, key, value, relative_bias = (self.split_heads(vector) for vector in [query, key, value, relative_bias])
    
    # r_w_bias and r_r_bias are of shape [1, num_heads, 1, head_size], only analyze the differences in heads and head_size
    AD = tf.matmul(query + self.r_w_bias,
                   key,
                   transpose_b = True)
    BD = tf.matmul(query + self.r_r_bias,
                   relative_bias,
                   transpose_b = True)
    # query * relative_bias is the relative analysis across multiple sequences
    # AD, BD e [batch_size, num_heads, sequence_size, sequence_size]
    BD = self.rel_shift(BD)
    logits = AD + BD
    logits /= self.head_size
    if self.unidirectional:
      bias = self.build_bias(self.input_size)
      # the memory, for all input sequences, should be available without interruption
      # therefore, a zeros vector is concatenated to the left of the bias
      # therefore, for the nth input sequence, the bias is [0 * memory_size, 0 * n, 1e-9 for the rest]
      bias = tf.concat([tf.zeros([1, 1, self.input_size, self.memory_size]), bias],
                       axis = -1)
      logits += bias
    if mask is not None:
      mask = tf.expand_dims(mask,
                            axis = 1)
      mask = tf.expand_dims(mask,
                            axis = 1)
      mask *= -1e9
      logits += mask
    weights = tf.nn.softmax(logits,
                            name = 'attention_weights')
    if self.adaptive_span:
      weights = self.adaptive_span_layer(weights,
                                         attention_output = query)
    
    output = tf.matmul(weights,
                       value)
    output = tf.transpose(output,
                          [0, 2, 1, 3])
    output = tf.reshape(output,
                        [tf.shape(output)[0],
                         tf.shape(output)[1],
                         self.num_heads * self.head_size])
    return self.output_dense(output)
    
  def rel_shift(self, x):
    x_shape = tf.shape(x)
    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
    # FLAGS: should this be transpose?
    x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2]])
    # FLAGS: x = x[:,:,1:]
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_shape)
    return x
    
class ReZero_Bias(tf.keras.layers.Layer):
  # the ReZero algorithm vectors, as specified in arXiv:2003.04887
  def __init__(self, layers,
               use_ffd,
               **kwargs):
    super(ReZero_Bias,
          self).__init__(**kwargs)
    self.layers = layers
    self.use_ffd = use_ffd
    
  def build(self, input_shape):
    self.r_zero_attention = self.add_weight('r_zero_attention',
                                            shape = [self.layers],
                                            initializer = tf.random_uniform_initializer(minval = -0.01, maxval = 0.01), # zeros
                                            dtype = tf.float32)
    if self.use_ffd:
      self.r_zero_ffd = self.add_weight('r_zero_ffd',
                                        shape = [self.layers],
                                        initializer = tf.random_uniform_initializer(minval = -0.01, maxval = 0.01), # zeros
                                        dtype = tf.float32)
                                            
  def call(self, inputs):
    if self.use_ffd:
      return self.r_zero_attention, self.r_zero_ffd
    else:
      return self.r_zero_attention

class Stop_Feature(tf.keras.layers.Layer):
  # the stop-feature function was initially proposed in arXiv:1804.00946
  # given an vector of shape [batch_size, sequence_size, hidden_size],
  # a stop-feature, of shape [batch_size, sequence_size,  1], is concatenated
  # the stop-feature is a smooth strictly-increasing function e [0, 1]
  # the authors of arXiv:1804.00946 describe it as a "temporal stamp to stick to each time step of a sequence"
  def __init__(self,
               **kwargs):
    super(Stop_Feature,
          self).__init__(**kwargs)
    
  def build(self, input_shape):
    pass
    
  def call(self, inputs):
    # FLAGS: tanh and exp
    batch_size, sequence_size = tf.shape(inputs)[0], tf.shape(inputs)[1]
    sequence_size = tf.cast(sequence_size,
                            dtype = tf.float32)
    stop_feature = tf.range(sequence_size,
                            dtype = tf.float32) / sequence_size
    stop_feature = tf.reshape(stop_feature,
                              [1, -1, 1])
    stop_feature = tf.tile(stop_feature,
                           [batch_size, 1, 1])
    inputs = tf.concat([inputs, stop_feature],
                       axis = -1)
    return inputs

class Timing_Signal(tf.keras.layers.Layer):
  # unlike Positional_Embedding, the Timing_Signal embeds position in a vector with a vectir that increases in a structures manner
  # the reason that Positional_Embedding outperforms the Timing_Signal might be because the Positional_Embedding, due to its trainable nature
  # can be tracked across a theoretically infinite number of layers
  # the Timing_Signal, on the other hand, is lost across layers
  def __init__(self, min_timescale = 1.0,
               max_timescale = 1.0e4,
               start_index = 0,
               **kwargs):
    super(Timing_Signal,
          self).__init__(**kwargs)
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale
    self.start_index = start_index
    
  def build(self, input_shape):
    self.hidden_size = input_shape[-1]
    
  def call(self, inputs):
    # the examples assume default parameters above, hidden_size = 8 and sequence_size = 2
    sequence_size = tf.shape(inputs)[1]
    position = tf.cast(tf.range(sequence_size) + self.start_index,
                       dtype = tf.float32)
    # position = [0.0, 1.0]
    # position e [sequence_size]
    num_timescales = self.hidden_size // 2
    # num_timescales = 4
    log_timescale_increment = (log_fn(float(self.max_timescale) / float(self.min_timescale)) / tf.maximum(tf.cast(num_timescales,
                                                                                                                  dtype = tf.float32) - 1,
                                                                                                          1))
    # log_timescale = 2.3026
    inv_timescales = self.min_timescale * tf.exp(tf.cast(tf.range(num_timescales),
                                                         dtype = tf.float32) * - log_timescale_increment)
    # inv_timescales = [1.0, 0.1, 0.01, 0.001]
    # inv_timescales e [hidden_size // 2]
    scaled_time = tf.expand_dims(position,
                                 axis = 1) * tf.expand_dims(inv_timescales,
                                                            axis = 0)
    # scaled = [[1.0, 0.1, 0.01, 0.001], [2.0, 1.1 1.01, 1.001]]
    # scaled e [sequence_size, hidden_size // 2]
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)],
                       axis = 1)
    # signal = [[0.84, 0.1, 0.01, 0.001, 0.54, 0.995, 0.99995, 0.9999995]
    #           [0.91, 0.89, 0.846, 0.842, -0.416, 0.453, 0.531, 0.53946056]]
    # signal e [sequence_size, hidden_size]
    signal = tf.pad(signal,
                    [[0, 0], [0, tf.math.mod(self.hidden_size,
                                             2)]])
    # signal [1, sequence_size, hidden_size] is automatically broadcasted to batch_size
    return tf.reshape(signal,
                      [1, sequence_size, self.hidden_size]) + inputs
                        
class Transformer_FFD(tf.keras.layers.Layer):
  # the FFD as specified in arXiv:1706.03762
  # quote: "FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2"
  # the authors specify two seperate feed forward layers, one after another, with a ReLU non-linear function between the two
  # the input size of the FFD, and the output size of the FFD, are of the same size
  # this is because the output and input of the FFD must be of identical size due to the residual connection
  # the dimension of the first FFD is specifed as 4 times the input size
  # it is possible to specify the filter size explicitly
  # BERT (arXiv:1810.04805) specified the activation function should be GeLU, not ReLU
  # the default activation is set to ReLU but can be changed to any function
  def __init__(self, activation = tf.nn.relu,
               filter_size = None,
               **kwargs):
    super(Transformer_FFD,
          self).__init__(**kwargs)
    self.activation = activation
    self.filter_size = filter_size
    # arXiv:1807.03819 suggests using separable convolution
    
  def build(self, input_shape):
    self.hidden_size = input_shape.as_list()[-1]
    if self.filter_size is None:
      self.filter_size = self.hidden_size * 4
    self.ffd_1 = tf.keras.layers.Dense(self.filter_size,
                                       activation = self.activation)
    self.ffd_2 = tf.keras.layers.Dense(self.hidden_size)
    
  def call(self, inputs,
           training,
           dropout_rate):
    ffd_1 = self.ffd_1(inputs)
    # dropout is applied in the FFD
    # dropout_rate and training are sent through in the call function
    # dropout_rate and training can be placeholders to allow this to be dynamic
    ffd_1 = tf.keras.layers.Dropout(dropout_rate)(ffd_1,
                                                  training = training)
    return self.ffd_2(ffd_1)
    
class Transformer_Gating(tf.keras.layers.Layer):
  # the gating function was introduced in arXiv:1910.06764
  # Transformers, in this case, were used for reinforcement learning rather then NLP
  # the authors found that the gating function was necessary to stabilize transformer learning for RL
  def __init__(self, gating,
               use_relu = True,
               **kwargs):
    super(Transformer_Gating,
          self).__init__(**kwargs)
    self.gating = gating
    self.use_relu = use_relu
    # if True, ACT = ReLU, else GeLU
    assert self.assert_gating()
    
  def assert_gating(self):
    if self.gating == 'residual':
      # output = ACT(y) + x
      # similar to traditional Tr-I seen in transformer.py, but includes the non-linear activaiton function
      # this is due to the fact that, without the activation function, there are two linear functions performed iteratively
      return True
    elif self.gating == 'input':
      # sigmoid(dense(x))*x + y
      return True
    elif self.gating == 'output':
      # x + sigmoid(dense(y))*y
      return True
    elif self.gating == 'highway':
      # sigmoid(dense(x)) * x + (1 - sigmoid(dense(x))) * y
      return True
    elif self.gating == 'sigmoid-tanh':
      # x + sigmoid(dense(x)) * tanh(dense(y))
      return True
    elif self.gating == 'gru':
      # r = sigmoid(dense(y) + dense(x))
      # z = sigmoid(dense(y) + dense(x))
      # h = tanh(dense(y) + dense(x * r))
      # output = (1 - z) * x + z * h
      return True
    return False
    
  def build(self, input_shape):
    self.hidden_size = input_shape[0][-1]
    if self.gating == 'input':
      self.layer = tf.keras.layers.Dense(self.hidden_size,
                                         activation = tf.identity,
                                         use_bias = False)
    elif self.gating == 'output':
      self.layer = tf.keras.layers.Dense(self.hidden_size,
                                         activation = tf.identity,
                                         bias_initializer = tf.initializers.constant(-1.0))
    elif self.gating == 'highway':
      self.layer = tf.keras.layers.Dense(self.hidden_size,
                                         activation = tf.identity)
    elif self.gating == 'sigmoid-tanh':
      self.layer = tf.keras.layers.Dense(self.hidden_size,
                                         activation = tf.identity,
                                         bias_initializer = tf.initializers.constant(-1.0))
      self.tanh_layer = tf.keras.layers.Dense(self.hidden_size,
                                              activation = tf.nn.tanh)
    elif self.gating == 'gru':
      self.x_layer = tf.keras.layers.Dense(self.hidden_size * 2,
                                           activation = tf.identity,
                                           use_bias = False)
      self.y_layer = tf.keras.layers.Dense(self.hidden_size * 3,
                                           activation = tf.identity,
                                           use_bias = False)
      self.r_x_layer = tf.keras.layers.Dense(self.hidden_size,
                                             activation = tf.identity,
                                             use_bias = False)
      self.z_bias = Bias_Add(self.hidden_size,
                             initializer = tf.constant_initializer(-1.0))
                                              
  def call(self, inputs):
    x, y = inputs
    activation = tf.nn.relu if self.use_relu else gelu
    y = activation(y)
    
    if self.gating == 'residual':
      return x + y
    elif self.gating == 'input':
      return tf.multiply(sigmoid_cutoff(self.layer(x)),
                         x) + y
    elif self.gating == 'output':
      return x + tf.multiply(sigmoid_cutoff(self.layer(x)),
                             y)
    elif self.gating == 'highway':
      highway = self.layer(x)
      return tf.multiply(sigmoid_cutoff(highway), 
                         x) + tf.multiply(1 - sigmoid_cutoff(highway),
                                          y)
    elif self.gating == 'sigmoid-tanh':
      return x + tf.multiply(sigmoid_cutoff(self.layer(x)),
                             tf.nn.tanh(self.tanh_layer(y)))
    elif self.gating == 'gru':
      batch_size, sequence_size = tf.shape(y)[0], tf.shape(y)[1]
      a = self.y_layer(y)
      b = self.x_layer(x)
      a = tf.reshape(a,
                     [batch_size, sequence_size, 3, self.hidden_size])
      b = tf.reshape(b,
                     [batch_size, sequence_size, 2, self.hidden_size])
      r = sigmoid_cutoff(a[:,:,0] + b[:,:,0])
      z = sigmoid_cutoff(self.z_bias(a[:,:,1] + b[:,:,1]))
      h = tf.nn.tanh(a[:,:,2] + self.r_x_layer(tf.multiply(r,
                                                           x)))
      return tf.multiply(1 - z, 
                         x) + tf.multiply(z, 
                                          h)
                                          
if __name__ == '__main__':
  inputs = tf.placeholder(tf.float32, [32, 100, 128])
  depthwise_attention = Multihead_Attention(head_size = 64, num_heads = 8, unidirectional = False)
  depthwise = depthwise_attention(inputs[:,:10], False, 0.0, key_value = inputs)
  print(inputs)
  print(depthwise)
  exit()
  attention = Multihead_Attention(head_size = 64, num_heads = 8, unidirectional = False)
  output = attention(inputs, False, 0.0, key_value = depthwise)
  print(output)