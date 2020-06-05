from __future__ import absolute_import, division, print_function, unicode_literals

from functools import partial

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# this should stop logging
import sys
import numpy as np
# %tensorflow_version 1.x
import tensorflow as tf
import time
import math
try:
  import psutil
except:
  pass

tf.random.set_random_seed(time.time())

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--BATCH_SIZE', default = 16, help = 'The per-GPU, per-iteration Batch Size', type = int)
parser.add_argument('--EPOCHS', default = 50, help = 'The number of epochs', type = int)
parser.add_argument('--ITERATIONS', default = 512, help = 'The number of iterations per epoch', type = int)
parser.add_argument('--LOAD_PREVIOUS_MODEL', default = False, help = 'Whether to load a previously saved model', type = bool)
parser.add_argument('--MODEL', default = 'transformer-xl', help = 'The model to be used', type = str)
parser.add_argument('--WARMUP_ITERATIONS', default = 5120, help = 'The number of warmup iterations', type = int)
parser.add_argument('--SEQ_LEN', default = 256, help = 'The sequence size used for language modelling', type = int)
parser.add_argument('--ADAPTIVE_SPAN', default = False, help = 'Whether to use an adaptive span', type = bool)
parser.add_argument('--ADAPTIVE_SPAN_HYPERPARAMETER', default = 0.001, help = 'The hyperparameter for the adaptive span, relating to loss', type = float)
parser.add_argument('--DYNAMIC_ADAPTIVE_SPAN', default = False, help ='Whether to use a dynamic adaptive span', type = bool)
parser.add_argument('--HEAD_SIZE', default = 64, help = 'The size of each attention head', type = int)
parser.add_argument('--HIDDEN_SIZE', default = 512, help = 'The hidden size of the model', type = int)
parser.add_argument('--LAYERS', default = 4, help = 'The number of layers of the model', type = int)
parser.add_argument('--NUM_HEADS', default = 8, help = 'The number of heads in each MHA', type = int)
parser.add_argument('--MEMORY_LENGTH', default = 256, help = 'The length of the XL Memory', type = int)
parser.add_argument('--USE_RELU', default = False, help = 'Whether to Use an ReLU or GELU', type = bool)
parser.add_argument('--CACHE_MEMORY', default = 'before_attention', help = 'When is memory cached [before_attention, after_attention, after_final_attention]', type = str)
parser.add_argument('--N_EXPERTS', default = 15, help = 'Number of experts in an MoS', type = int)
parser.add_argument('--USE_MOS', default = False, help = 'Whether to use an MoS rather then a softmax classifier', type = bool)
parser.add_argument('--CLIP_VALUE', default = 1e-1, help = 'The value to clip the gradient norm', type = float)
parser.add_argument('--DROPOUT_RATE', default = 1e-1, help = 'The dropout rate', type = float)
parser.add_argument('--INITIAL_LEARNING_RATE', default = 1e-6, help = 'The initial learning rate', type = float)
parser.add_argument('--STABLE_LEARNING_RATE', default = 5e-4, help = 'The stable learning rate', type = float)
parser.add_argument('--FINAL_LEARNING_RATE', default = 1e-6, help = 'The final learning rate', type = float)
parser.add_argument('--WARMUP_RATE', default = 'linear-warmup', help = 'The warmup learning schedule', type = str)
parser.add_argument('--DECAY_RATE', default = 'cosine', help = 'The decay learning schedule', type = str)
parser.add_argument('--RNN_CELL', default = 'lstm', help = 'The RNN Cell', type = str)
parser.add_argument('--TOKENIZATION_SCHEME', default = 'character', help = 'The tokenization scheme of the data [character, 1000-bpe]', type = str)

args = parser.parse_args()
    
def clip_by_norm(grad,
                 value):
  if grad is None:
    return None
  else:
    return tf.clip_by_norm(grad,
                           value)

class Writer():
  def __init__(self, name = 'output'):
    i = 1
    while os.path.exists('{}-{}.txt'.format(name,
                                            i)):
      i += 1
    self.file = open('{}-{}.txt'.format(name,
                                        i),
                     'w')
                     
  def print_line(self, line):
    self.file.write('{} \n'.format(line))
    print(line)
    
  def close_file(self):
    self.file.close()

class Learning_Rate():
  def __init__(self, initial_learning_rate,
               stable_learning_rate,
               final_learning_rate,
               warmup_rate,
               decay_rate,
               warmup_steps,
               decay_steps,
               writer,
               step_size = 1000):
    self.initial_learning_rate = initial_learning_rate
    # the initial learning rate, at the start of the warmup steps
    # if the warmup steps is set to 0 or warmup_rate == 'constant', initial_learning_rate == stable_learning_rate
    self.stable_learning_rate = stable_learning_rate
    # the learning rate at the end of warmup steps, and the beginning of the decay steps
    self.final_learning_rate = final_learning_rate
    # the final learning rate
    # if decay_rate == 'constant', stable_learning_rate == final_learning_rate
    self.warmup_rate = warmup_rate
    # warmup refers to where the learning rate inceases over iterations
    # 'constant', 'linear-warmup'
    self.decay_rate = decay_rate
    # learning rate decreases over iterations
    # 'constant' 'linear-decay' 'polynomial' 'cosine' 'cyclic-triangular' 'cyclic-triangular2' 'cyclic-exp'
    self.warmup_steps = warmup_steps 
    self.decay_steps = decay_steps 
    if self.decay_steps <= 0:
      self.decay_steps = 1
    self.step_size = step_size
    # only used for cyclic learning rates
    
    if self.warmup_rate == 'inverse':
      self.decay_rate = 'inverse'
      writer.print_line('Learning Rate Schedule: Inverse')
      writer.print_line('Warmup Steps: {}'.format(self.warmup_steps))
      from learning_rate import Inverse_Square_Root_Learning_rate
      self.learning_rate = Inverse_Square_Root_Learning_rate(learning_rate = self.initial_learning_rate,
                                                             warmup_steps = self.warmup_steps)
    else:
      writer.print_line('Initial Learning Schedule: {}'.format(self.warmup_rate))
      if self.warmup_rate == 'linear-warmup':
        writer.print_line('Warmup Learning Rate: {}'.format(self.initial_learning_rate))
      writer.print_line('Warmup Steps: {}'.format(self.warmup_steps))
      writer.print_line('Stable Learning Rate: {}'.format(self.stable_learning_rate))
      writer.print_line('Final Learning Schedule: {}'.format(self.decay_rate))
      if self.decay_rate != 'linear-decay':
        writer.print_line('Final Learning Rate: {}'.format(self.final_learning_rate))
        if 'cyclic' in self.decay_rate:
          writer.print_line('Cyclic Step Size: {}'.format(self.step_size))
      writer.print_line('Decay Steps: {}'.format(self.decay_steps))
      writer.print_line('Total Number of Iterations: {}'.format(self.warmup_steps + self.decay_steps))
    
      self.iteration = 0
    
      from learning_rate import Linear_Learning_Rate, Polynomial_Decay_Learning_Rate, Cosine_Learning_Rate, Cyclic_Learning_Rate
      if self.warmup_rate == 'constant':
        self.warmup_learning_rate = self.initial_learning_rate
        assert self.initial_learning_rate == self.stable_learning_rate
        # if the learning rate is constant, then warmup_learning_rate is set to a float, not a class from learning_rate.py
      elif self.warmup_rate == 'linear-warmup':
        self.warmup_learning_rate = Linear_Learning_Rate(learning_rate = self.stable_learning_rate - self.initial_learning_rate,
                                                         final_learning_rate = 0,
                                                         decay_steps = 0,
                                                         warmup_steps = self.warmup_steps,
                                                         stable_steps = 0)
      
      if self.decay_rate == 'constant':
        self.decay_learning_rate = self.stable_learning_rate
        assert self.stable_learning_rate == self.final_learning_rate
      elif self.decay_rate == 'linear-decay':
        self.decay_learning_rate = Linear_Learning_Rate(learning_rate = self.stable_learning_rate,
                                                        final_learning_rate = self.final_learning_rate,
                                                        decay_steps = self.decay_steps,
                                                        warmup_steps = 0,
                                                        stable_steps = 0)
      elif self.decay_rate == 'polynomial':
        self.decay_learning_rate = Polynomial_Decay_Learning_Rate(learning_rate = self.stable_learning_rate,
                                                                  max_epochs = self.decay_steps,
                                                                  min_learning_rate = self.final_learning_rate)
      elif self.decay_rate == 'cosine':
        self.decay_learning_rate = Cosine_Learning_Rate(learning_rate = self.stable_learning_rate,
                                                        decay_steps = self.decay_steps,
                                                        min_learning_rate = self.final_learning_rate)
      else:
        if self.decay_rate == 'cyclic-triangular':
          mode = 'triangular'
        elif self.decay_rate == 'cyclic-triangular2':
          mode = 'triangular2'
        elif self.decay_rate == 'cyclic-exp':
          mode = 'exp_range'
        self.decay_learning_rate = Cyclic_Learning_Rate(learning_rate = self.stable_learning_rate,
                                                        step_size = self.step_size,
                                                        mode = mode,
                                                        warmup_steps = 0,
                                                        stable_steps = 100)
                                                      
  def call(self):
    if self.warmup_rate == 'inverse':
      return self.learning_rate()
    self.iteration += 1
    if self.iteration < self.warmup_steps:
      if self.warmup_rate == 'constant':
        return self.warmup_learning_rate
      else:
        return self.warmup_learning_rate()
    else:
      if self.decay_rate == 'constant':
        return self.decay_learning_rate
      else:
        return self.decay_learning_rate()
        
  def add_iteration(self):
    self.iteration += 1
    if self.iteration < self.warmup_steps:
      if self.warmup_rate != 'constant':
        self.warmup_learning_rate.iteration += 1
    else:
      if self.decay_rate != 'constant':
        self.decay_learning_rate.iteration += 1

def flatten(tensor):
  final_dim = tensor.shape.as_list()[-1]
  tensor = tf.reshape(tensor,
                      [-1, final_dim])
  return tensor

def average_gradients(tower_grads_fn,
                      static = False):
  # average gradients must work over both tensors (during building) and IndexSlices (dynamically)
  # if static is True, graph is building
  # else, gradients must be calculated dynamically
  grads_fn = []
  for grad in range(len(tower_grads_fn[0])):
    if static:
      name = tower_grads_fn[0][grad].name
      expand_dims = lambda x: tf.expand_dims(x,
                                             axis = 0)
      concatenate = lambda x: tf.concat(x,
                                        axis = 0)
      mean = lambda x: tf.reduce_mean(x, 
                                      axis = 0,
                                      name = name[:-2]+'/mean')
    else:
      expand_dims = lambda x: np.expand_dims(x,
                                             axis = 0)
      concatenate = lambda x: np.concatenate(x,
                                             axis = 0)
      mean = lambda x: np.mean(x,
                               axis = 0)
    average_grad = []
    for tower in range(len(tower_grads_fn)):
      average_grad.append(expand_dims(tower_grads_fn[tower][grad]))
    average_grad = concatenate(average_grad)
    average_grad = mean(average_grad)
    grads_fn.append(average_grad)
  return grads_fn

def run(args):
  writer = Writer()
  writer.print_line(os.getcwd())
  writer.print_line('tensorflow version: {}'.format(tf.__version__))

  BATCH_SIZE = args.BATCH_SIZE
  EPOCHS = args.EPOCHS
  EVALUATE = False
  ITERATIONS = args.ITERATIONS
  LOAD_PREVIOUS_MODEL = args.LOAD_PREVIOUS_MODEL
  MODEL = args.MODEL
  
  RUN_PREDICTIONS = False
  TEST_BATCH_SIZE = 5
  TEST_ITERATIONS = 100
  VALID_ITERATIONS = 100
  WARMUP_ITERATIONS = args.WARMUP_ITERATIONS
  
  writer.print_line('Epochs: {}'.format(EPOCHS))
  writer.print_line('Iterations per Epoch: {}'.format(ITERATIONS))
  writer.print_line('Valid Iterations per Epoch: {}'.format(VALID_ITERATIONS))
  writer.print_line('Test Iterations: {}'.format(TEST_ITERATIONS))
    
  TOP_K = 5
    
  try:
    NUM_GPUS = len(tf.config.experimental.list_physical_devices('GPU'))
    assert NUM_GPUS != 0
    writer.print_line('Number of GPUs (set dynamically): {}'.format(NUM_GPUS))
  except:
    NUM_GPUS = 1
    writer.print_line('Number of GPUs (set manually): {}'.format(NUM_GPUS))
    
  GLOBAL_BATCH_SIZE = NUM_GPUS * BATCH_SIZE
  
  SEQ_LEN = args.SEQ_LEN
  
  writer.print_line('Batch Size per GPU: {}'.format(BATCH_SIZE))
  writer.print_line('Global Batch Size: {}'.format(GLOBAL_BATCH_SIZE))
  writer.print_line('Sequence Size per Batch: {}'.format(SEQ_LEN))
  
  TOKENIZATION_SCHEME = args.TOKENIZATION_SCHEME
  from python_feeder import Python_Feeder
  feeder = Python_Feeder(batch_size = GLOBAL_BATCH_SIZE,
                         tokenization_scheme = TOKENIZATION_SCHEME)
  if TOKENIZATION_SCHEME == 'character':
    VOCAB_SIZE = 256
  else:
    VOCAB_SIZE = 1000
  DOWNSTREAM_TASK = False
  writer.print_line('Unidirectional Source Code Statistical Analysis')
  writer.print_line('Python Programming Language')
  writer.print_line('Tokenization Scheme: {}'.format(TOKENIZATION_SCHEME))
    
  dropout_rate_fn = tf.placeholder(tf.float32)
  training_fn = tf.placeholder(tf.bool)
  
  ADAPTIVE_SPAN = args.ADAPTIVE_SPAN
  ADAPTIVE_SPAN_HYPERPARAMETER = args.ADAPTIVE_SPAN_HYPERPARAMETER
  DYNAMIC_ADAPTIVE_SPAN = args.DYNAMIC_ADAPTIVE_SPAN
  HEAD_SIZE = args.HEAD_SIZE
  HIDDEN_SIZE = args.HIDDEN_SIZE
  LAYERS = args.LAYERS
  NUM_HEADS = args.NUM_HEADS
  MEMORY_LENGTH = args.MEMORY_LENGTH
  
  USE_RELU = args.USE_RELU

  if MODEL == 'transformer-xl':
    CACHE_MEMORY = args.CACHE_MEMORY
    from transformer_xl import Transformer_XL
    model = Transformer_XL(adaptive_span = ADAPTIVE_SPAN,
                           batch_size = None,
                           cache_memory = CACHE_MEMORY,
                           dropout_rate = dropout_rate_fn,
                           embed_size = HIDDEN_SIZE,
                           head_size = HEAD_SIZE,
                           hidden_size = HIDDEN_SIZE,
                           input_vocab_size = VOCAB_SIZE,
                           layers = LAYERS,
                           max_length = None,
                           max_memory_length = MEMORY_LENGTH,
                           memory_length = None,
                           num_heads = NUM_HEADS,
                           output_vocab_size = None,
                           seq_len = None,
                           training = training_fn,
                           unidirectional = True,
                           use_memory = True,
                           use_prev = False,
                           use_positional_timing = False,
                           use_relu = USE_RELU,
                           use_signal_timing = True)
  elif MODEL == 'rnn':
    RNN_CELL = args.RNN_CELL
    from lstm import LSTM
    
    N_EXPERTS = args.N_EXPERTS
    USE_MOS = args.USE_MOS
    
    model = LSTM(dropout_rate = dropout_rate_fn,
                 embed_size = HIDDEN_SIZE,
                 memory_size = HIDDEN_SIZE,
                 input_vocab_size = VOCAB_SIZE,
                 layers = LAYERS,
                 output_vocab_size = None,
                 rnn_cell = RNN_CELL,
                 training = training_fn,
                 n_experts = N_EXPERTS,
                 use_mos = USE_MOS)
    
  writer.print_line(MODEL)
  writer.print_line('No. of layers: {}'.format(LAYERS))
  writer.print_line('Hidden size: {}'.format(HIDDEN_SIZE))
  writer.print_line('Vocabulary size: {}'.format(VOCAB_SIZE))
  if MODEL == 'rnn':
    writer.print_line('RNN Cell: {}'.format(RNN_CELL))
    if USE_MOS:
      writer.print_line('MoS Used')
      writer.print_line('Number of Experts: {}'.format(N_EXPERTS))
  else:
    writer.print_line('Number of Heads Per Attention Mechanism: {}'.format(NUM_HEADS))
    if USE_RELU:
      writer.print_line('The ReLU activation function is used')
    else:
      writer.print_line('The GeLU activation function is used')
    if ADAPTIVE_SPAN:
      writer.print_line('Adaptive Span Is Used')
      writer.print_line('Adaptive Span Hyperparameter: {}'.format(ADAPTIVE_SPAN_HYPERPARAMETER))
      if DYNAMIC_ADAPTIVE_SPAN and (MODEL == 'transformer' or MODEL == 'acg-transformer'):
        writer.print_line('Dynamic Adaptive Span')
      else:
        writer.print_line('Static Adaptive Span')
    writer.print_line('XL Memory Length: {}'.format(MEMORY_LENGTH))
    writer.print_line('XL Memory Cache Method: {}'.format(CACHE_MEMORY))

  keras_model = model.model
  
  inputs_fn = tf.placeholder(tf.int32,
                             [None, None],
                             name = 'inputs')
  # (BATCH_SIZE, SEQ_LEN), INT32
  targets_fn = tf.placeholder(tf.int32,
                              [None, None],
                              name = 'targets')
  # (BATCH_SIZE, SEQ_LEN), INT32
  (tower_inputs_fn, tower_targets_fn) = (tf.split(tensor,
                                                  num_or_size_splits = NUM_GPUS,
                                                  axis = 0) for tensor in [inputs_fn,
                                                                           targets_fn])
  if MODEL == 'transformer-xl':
    memory_fn = tf.placeholder(tf.float32,
                               [None, LAYERS, None, HIDDEN_SIZE],
                               name = 'memory')
    # (BATCH_SIZE, LAYERS, SEQ_LEN, HIDDEN_SIZE), FLOAT32
    tower_memory_fn = tf.split(memory_fn, 
                               num_or_size_splits = NUM_GPUS,
                               axis = 0)
  elif MODEL == 'rnn':
    if RNN_CELL == 'lstm':
      state_fn = tf.placeholder(tf.float32,
                                [None, LAYERS, 2, HIDDEN_SIZE],
                                name = 'state')
    else:
      state_fn = tf.placeholder(tf.float32,
                                [None, LAYERS, HIDDEN_SIZE],
                                name = 'state')
    tower_state_fn = tf.split(state_fn,
                              num_or_size_splits = NUM_GPUS,
                              axis = 0)
  # the inputs, targets, memory and state are all split across the different GPUs
  # the tensors are split across the batch dimension
  # this is done as a part of the distribution strategy
  
  learning_rate_fn = tf.placeholder(tf.float32,
                                    name = 'learning-rate')
  # the learning rate is changed dynamically, so is set as a placeholder
  optimizer_fn = tf.train.AdamOptimizer(learning_rate = learning_rate_fn,
                                        beta1 = 0.9,
                                        beta2 = 0.999,
                                        epsilon = 1e-8)
  
  tower_grads_fn = []
  tower_compressive_grads_fn = []
  tower_loss_fn = []
  tower_compressive_loss_fn = []
  tower_accuracy_fn = []
  tower_sequential_accuracy_fn = []
  tower_top_k_accuracy_fn = []
  # the distribution strategy is implemented here
  # an identical model is loaded onto each GPU and different batch is run on each GPU
  # the gradients are averaged across each GPU and applied to the model
  # in essence, the GPUs perform the computationally expensive aspects independentally
  # the computationally inexpensive aspect, the averaging and application of the gradients, is done outside of the distribution strategy
  for i in range(NUM_GPUS):
    with tf.device('/gpu:{}'.format(i)):
      # for each GPU, a model is built on the device
      if MODEL == 'rnn':
        tower_logits_fn, tower_new_state_fn = keras_model((tower_inputs_fn[i],
                                                           tower_state_fn[i]))
      elif MODEL == 'transformer-xl':
        tower_logits_fn, tower_new_memory_fn = keras_model((tower_inputs_fn[i],
                                                            tower_memory_fn[i]))
      
      if i == 0:
        trainable_variables = sum([np.prod(variable.shape) for variable in keras_model.trainable_variables])
        writer.print_line('Trainable Variables: {}'.format(trainable_variables))
        
      if i == 0:
        logits_fn = tower_logits_fn
        # for the 0'th GPU, the prediction and the memory tensors are set as predict_fn, new_memory_fn and new_compressed_memory_fn
        # otherwise, the tensors are concatenated
        predict_fn = tf.argmax(tower_logits_fn,
                               axis = -1,
                               output_type = tf.int32)
        if MODEL == 'transformer-xl':
          new_memory_fn = tower_new_memory_fn
        elif MODEL == 'rnn':
          new_state_fn = tower_new_state_fn
      else:
        logits_fn = tf.concat([logits_fn, tower_logits_fn],
                              axis = 0)
        predict_fn = tf.concat([predict_fn, tf.argmax(tower_logits_fn,
                                                      axis = -1,
                                                      output_type = tf.int32)],
                               axis = 0)
        if MODEL == 'transformer-xl':
          new_memory_fn = tf.concat([new_memory_fn, tower_memory_fn],
                                    axis = 0)
        elif MODEL == 'rnn':
          new_state_fn = tf.concat([new_state_fn, tower_new_state_fn],
                                   axis = 0)
      crossentropy_fn = tf.keras.losses.sparse_categorical_crossentropy(tower_targets_fn[i],
                                                                        tower_logits_fn)
      crossentropy_fn = tf.reduce_mean(crossentropy_fn)
      
      if ADAPTIVE_SPAN:
        adaptive_span_variables = [variable for variable in keras_model.trainable_variables if 'current_val' in variable.name]
        
        adaptive_span_variables = [tf.reshape(variable, shape = [-1]) for variable in adaptive_span_variables]
        for var in range(len(adaptive_span_variables)):
          if adaptive_span_variables[var].shape[0] != HEAD_SIZE:
            adaptive_span_variables[var] = tf.tile(adaptive_span_variables[var], [HEAD_SIZE // adaptive_span_variables[var].shape[0]]) / HEAD_SIZE
        adaptive_span_loss = tf.reduce_mean(adaptive_span_variables) / NUM_HEADS
        crossentropy_fn += adaptive_span_loss * ADAPTIVE_SPAN_HYPERPARAMETER
        
      tower_grads_fn.append(tf.gradients(crossentropy_fn,
                                         keras_model.trainable_variables))
      tower_loss_fn.append(crossentropy_fn)
      
      categorical_accuracy_fn = tf.keras.metrics.sparse_categorical_accuracy(tower_targets_fn[i],
                                                                             tower_logits_fn)
      categorical_accuracy_fn = tf.reduce_mean(categorical_accuracy_fn)
      tower_accuracy_fn.append(categorical_accuracy_fn)
      
      categorical_top_k_accuracy_fn = tf.keras.metrics.top_k_categorical_accuracy(flatten(tf.one_hot(tower_targets_fn[i],
                                                                                                     depth = VOCAB_SIZE,
                                                                                                     dtype = tf.float32)),
                                                                                  flatten(tower_logits_fn),
                                                                                  k = TOP_K)
      categorical_top_k_accuracy_fn = tf.reduce_mean(categorical_top_k_accuracy_fn)
      tower_top_k_accuracy_fn.append(categorical_top_k_accuracy_fn)
      # loss, accuracy and top_k accuracy are all calculcated per-tower
  
  loss_fn, loss_op_fn = tf.metrics.mean(tf.reduce_mean(tower_loss_fn),
                                        name = 'crossentropy')
  accuracy_fn, accuracy_op_fn = tf.metrics.mean(tf.reduce_mean(tower_accuracy_fn),
                                                name = 'accuracy')
  top_k_accuracy_fn, top_k_accuracy_op_fn = tf.metrics.mean(tf.reduce_mean(tower_top_k_accuracy_fn),
                                                            name = 'top_k_accuracy')
  recall_fn, recall_op_fn = tf.metrics.recall(tf.one_hot(targets_fn, 
                                                         depth = VOCAB_SIZE),
                                              logits_fn)
  precision_fn, precision_op_fn = tf.metrics.precision(tf.one_hot(targets_fn, 
                                                                  depth = VOCAB_SIZE),
                                                       logits_fn)
  
  if len(tower_grads_fn) == 1:
    grads_fn = tower_grads_fn[0]
  else:
    grads_fn = average_gradients(tower_grads_fn)
    # the gradients are averaged across GPU towers
  # the compressive transformer actually collects the gradients, across multiple runs, and averages them after 60,000 iterations
  
  CLIP_VALUE = args.CLIP_VALUE
  if CLIP_VALUE is not None:
    writer.print_line('Values to clip gradient norm: {}'.format(CLIP_VALUE))
    clipped_grads_fn = [clip_by_norm(grad,
                                     CLIP_VALUE) for grad in grads_fn]
    # multiple deep NLP papers have recommended the use of gradient clipping
    # compressive transformer states that gradient norm is clipped to 1e-1
    # megatron clips gradients to 1.0
  train_op_fn = optimizer_fn.apply_gradients(zip(clipped_grads_fn,
                                                 keras_model.trainable_variables))
  
  global_variables = tf.global_variables()
  # global variables are the trainable model variables
  local_variables = tf.local_variables()
  # the local variables are the variables related to the metrics
  crossentropy_variables = [v for v in local_variables if 'crossentropy/' in v.name]
  accuracy_variables = [v for v in local_variables if 'accuracy/' in v.name]
  top_k_accuracy_variables = [v for v in local_variables if 'top_k_accuracy/' in v.name]
  recall_variables = [v for v in local_variables if 'recall/' in v.name]
  precision_variables = [v for v in local_variables if 'precision/' in v.name]
  
  saver = tf.train.Saver()
  # the saver is used to save the session of the model
  
  DROPOUT_RATE = args.DROPOUT_RATE
  # dropout set 0.1 (except GELU dropout, set to 0.0) for some models
  writer.print_line('Constant Dropout: {}'.format(DROPOUT_RATE))
  INITIAL_LEARNING_RATE = args.INITIAL_LEARNING_RATE
  STABLE_LEARNING_RATE = args.STABLE_LEARNING_RATE
  FINAL_LEARNING_RATE = args.FINAL_LEARNING_RATE
  # the compressive transformer uses a linear warmup rate, but the iterations-per-optimization-scheme is increased from 1-4 in place of a decay rate
  WARMUP_RATE = args.WARMUP_RATE
  # 'constant' 'linear-warmup'
  DECAY_RATE = args.DECAY_RATE
  if WARMUP_RATE == 'constant':
    assert INITIAL_LEARNING_RATE == STABLE_LEARNING_RATE
  if DECAY_RATE == 'constant':
    assert STABLE_LEARNING_RATE == FINAL_LEARNING_RATE
  learning_rate = Learning_Rate(initial_learning_rate = INITIAL_LEARNING_RATE,
                                stable_learning_rate = STABLE_LEARNING_RATE,
                                final_learning_rate = FINAL_LEARNING_RATE,
                                warmup_rate = WARMUP_RATE, 
                                decay_rate = DECAY_RATE,
                                warmup_steps = WARMUP_ITERATIONS,
                                decay_steps = (EPOCHS * ITERATIONS - WARMUP_ITERATIONS),
                                writer = writer)
  
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                                          log_device_placement=False))
  sess.run(tf.variables_initializer(global_variables))
  sess.run(tf.variables_initializer(local_variables))
  
  metrics = {'train_loss': [],
             'test_loss': [],
             'time_per_epoch': []}
  
  if LOAD_PREVIOUS_MODEL:
    saver.restore(sess, 
                  '{}.ckpt'.format(MODEL))
  
  writer.print_line('# Time: {}'.format(time.asctime()))
  writer.print_line('')
  writer.print_line('')
  writer.print_line('')
  
  if not EVALUATE:
    # if the model is to be trained, not just tested
    for epoch in range(EPOCHS):
      start_time = time.time()
      
      if MODEL == 'transformer-xl':
        memory = np.zeros([BATCH_SIZE * NUM_GPUS, LAYERS, 0, HIDDEN_SIZE])
      elif MODEL == 'rnn':
        if RNN_CELL == 'lstm':
          state = np.zeros([BATCH_SIZE * NUM_GPUS, LAYERS, 2, HIDDEN_SIZE])
        else:
          state = np.zeros([BATCH_SIZE * NUM_GPUS, LAYERS, HIDDEN_SIZE])
      # the memory is initialized a the start of each epoch, where the sequence size is set to 0
  
      for iteration in range(ITERATIONS):
        data = feeder(sequence_size = SEQ_LEN)
        # the data from feeder is of shape [batch size, sequence size + 1]
        trainX, trainY = data[:,:-1], data[:,1:]
        
        fetches = [train_op_fn]
        
        fetches.extend([loss_op_fn, 
                        accuracy_op_fn, 
                        top_k_accuracy_op_fn, 
                        recall_op_fn, 
                        precision_op_fn])
        # the operations in fetches are currently all op_fn (except possibly grads_fn)
        # these operations, when run through sess, do not return any values
        feed_dict = {inputs_fn: trainX,
                     targets_fn: trainY,
                     dropout_rate_fn: DROPOUT_RATE,
                     training_fn: True}
        if MODEL == 'transformer-xl':
          fetches.append(new_memory_fn)
          feed_dict[memory_fn] = memory
        elif MODEL == 'rnn':
          fetches.append(new_state_fn)
          feed_dict[state_fn] = state
        feed_dict[learning_rate_fn] = learning_rate.call()
        
        fetched = sess.run(fetches,
                           feed_dict = feed_dict)
        
        if MODEL == 'transformer-xl':
          memory = fetched[-1]
        elif MODEL == 'rnn':
          state = fetched[-1]
        
      loss, accuracy, top_k_accuracy, recall, precision = sess.run([loss_fn, 
                                                                    accuracy_fn, 
                                                                    top_k_accuracy_fn, 
                                                                    recall_fn, 
                                                                    precision_fn])
      predict = sess.run(predict_fn,
                         feed_dict = feed_dict)
      train_loss = loss
      metrics['train_loss'].append(loss)
      # the values of the metrics are all extracted from the model
      # these are the average (loss, accuracy, top_k_accuracy, recall, precision) from the entire epoch
      assert not np.isnan(loss), 'Loss NaN at epoch {}'.format(epoch)
      writer.print_line('#' * 10 + ' Training Details ' + '#' * 10)
      writer.print_line('# Epoch {}'.format(epoch + 1))
      writer.print_line('# Loss: {:.10f}'.format(loss))
      # note that train loss on its own is a useless metric
      # it is only when used alongside the test loss that it becomes useful
      # this is done by measuring the overfitting metric, see below
      writer.print_line('# BPC: {:.10f}'.format(loss / math.log(2)))
      writer.print_line('# Perplexity: {:.10f}'.format(math.pow(np.e,
                                                       loss)))
      writer.print_line('# Accuracy: {:.10f}'.format(accuracy))
      writer.print_line('# Top-{}: {:.10f}'.format(TOP_K,
                                                   top_k_accuracy))
      writer.print_line('# Recall: {:.10f}'.format(recall))
      writer.print_line('# Precision: {:.10f}'.format(precision))
      # F1 score is defined as:
      f_1 = (2 * precision * recall) / (precision + recall)
      writer.print_line('# F1 Score: {:.10f}'.format(f_1))
      writer.print_line('# Training time in seconds: {}'.format(time.time() - start_time))
      # for Transformer, training time: 1.3-1.4sec per iteration
      try:
        writer.print_line(psutil.virtual_memory())
      except:
        writer.print_line(os.popen('free -t -m').readlines())
      writer.print_line('# Time: {}'.format(time.asctime()))
      writer.print_line(predict[0])
      writer.print_line(trainY[0])
      
      sess.run(tf.variables_initializer(crossentropy_variables))
      sess.run(tf.variables_initializer(accuracy_variables))
      sess.run(tf.variables_initializer(top_k_accuracy_variables))
      sess.run(tf.variables_initializer(recall_variables))
      sess.run(tf.variables_initializer(precision_variables))
      # the variables regarding the metrics must be re-initialized
      
      saver.save(sess, 
                 '{}.ckpt'.format(MODEL))
      
      if MODEL == 'transformer-xl':
        memory = np.zeros([BATCH_SIZE * NUM_GPUS, LAYERS, 0, HIDDEN_SIZE])
      elif MODEL == 'rnn':
        if RNN_CELL == 'lstm':
          state = np.zeros([BATCH_SIZE * NUM_GPUS, LAYERS, 2, HIDDEN_SIZE])
        else:
          state = np.zeros([BATCH_SIZE * NUM_GPUS, LAYERS, HIDDEN_SIZE])
        
      start_time = time.time()
      for iteration in range(VALID_ITERATIONS):
        data = feeder(task = 'valid',
                      sequence_size = SEQ_LEN)
        validX, validY = data[:,:-1], data[:,1:]
        fetches = [loss_op_fn, 
                   accuracy_op_fn, 
                   top_k_accuracy_op_fn, 
                   recall_op_fn, 
                   precision_op_fn]
        
        feed_dict = {inputs_fn: validX,
                     targets_fn: validY,
                     dropout_rate_fn: 0.0,
                     training_fn: False}
        if MODEL == 'transformer-xl':
          fetches.append(new_memory_fn)
          feed_dict[memory_fn] = memory
        elif MODEL == 'rnn':
          fetches.append(new_state_fn)
          feed_dict[state_fn] = state
          
        fetched = sess.run(fetches,
                           feed_dict = feed_dict)
        
        if MODEL == 'transformer-xl':
          if not DOWNSTREAM_TASK:
            memory = fetched[-1]
        elif MODEL == 'rnn':
          if not DOWNSTREAM_TASK:
            state = fetched[-1]
          
      loss, accuracy, top_k_accuracy, recall, precision = sess.run([loss_fn, 
                                                                    accuracy_fn, 
                                                                    top_k_accuracy_fn, 
                                                                    recall_fn, 
                                                                    precision_fn])
      metrics['test_loss'].append(loss)
      
      assert not np.isnan(loss), 'Loss NaN at epoch {}'.format(epoch)
      writer.print_line('#' * 10 + ' Valid Details ' + '#' * 10)
      writer.print_line('# Epoch {}'.format(epoch + 1))
      writer.print_line('# Loss: {:.10f}'.format(loss))
      writer.print_line('# BPC: {:.10f}'.format(loss / math.log(2)))
      writer.print_line('# Perplexity: {:.10f}'.format(math.pow(np.e,
                                                                loss)))
      writer.print_line('# Accuracy: {:.10f}'.format(accuracy))
      writer.print_line('# Top-{}: {:.10f}'.format(TOP_K,
                                                   top_k_accuracy))
      writer.print_line('# Recall: {:.10f}'.format(recall))
      writer.print_line('# Precision: {:.10f}'.format(precision))
      f_1 = (2 * precision * recall) / (precision + recall)
      writer.print_line('# F1 Score: {:.10f}'.format(f_1))
      writer.print_line('# Prediction time in seconds: {}'.format(time.time() - start_time))
      # for Transformer, prediction time: 0.38sec per iteration
      writer.print_line('# Overfitting Metric: {:.10f}'.format(loss / train_loss))
      # the higher the overfitting metric, the more the model has overfit
      writer.print_line('# Time: {}'.format(time.asctime()))
      writer.print_line('')
      writer.print_line('')
      writer.print_line('')
      
      sess.run(tf.variables_initializer(crossentropy_variables))
      sess.run(tf.variables_initializer(accuracy_variables))
      sess.run(tf.variables_initializer(top_k_accuracy_variables))
      sess.run(tf.variables_initializer(recall_variables))
      sess.run(tf.variables_initializer(precision_variables))
      
      metrics['time_per_epoch'].append(time.time() - start_time)
  
  if EVALUATE and not LOAD_PREVIOUS_MODEL:
    # if the code is simply for evaluation, and not for training
    # and the previous model has already been loaded, then the code here will load the model twice
    # so put in an if so the load is, at max, 1
    saver.restore(sess, 
                  '{}.ckpt'.format(MODEL))
  
  if MODEL == 'transformer-xl':
    memory = np.zeros([BATCH_SIZE * NUM_GPUS, LAYERS, 0, HIDDEN_SIZE])
  elif MODEL == 'rnn':
    if RNN_CELL == 'lstm':
      state = np.zeros([BATCH_SIZE * NUM_GPUS, LAYERS, 2, HIDDEN_SIZE])
    else:
      state = np.zeros([BATCH_SIZE * NUM_GPUS, LAYERS, HIDDEN_SIZE])
          
  for iteration in range(TEST_ITERATIONS):
    data = feeder(task = 'test',
                  sequence_size = SEQ_LEN)
    testX, testY = data[:,:-1], data[:,1:]
    
    fetches = [loss_op_fn, 
               accuracy_op_fn, 
               top_k_accuracy_op_fn, 
               recall_op_fn, 
               precision_op_fn]
    feed_dict = {inputs_fn: testX,
                 targets_fn: testY,
                 dropout_rate_fn: 0.0,
                 training_fn: False}
    if MODEL == 'transformer-xl':
      fetches.append(new_memory_fn)
      feed_dict[memory_fn] = memory
    elif MODEL == 'rnn':
      fetches.append(new_state_fn)
      feed_dict[state_fn] = state
    
    fetched = sess.run(fetches,
                       feed_dict = feed_dict)
    if MODEL == 'transformer-xl':
      memory = fetched[-1]
    elif MODEL == 'rnn':
      state = fetched[-1]
      
  loss, accuracy, top_k_accuracy, recall, precision = sess.run([loss_fn, 
                                                                accuracy_fn, 
                                                                top_k_accuracy_fn, 
                                                                recall_fn, 
                                                                precision_fn])
  assert not np.isnan(loss), 'Loss NaN at epoch {}'.format(epoch)
  writer.print_line('#' * 10 + ' Test Details ' + '#' * 10)
  writer.print_line('# Epoch {}'.format(epoch + 1))
  writer.print_line('# Loss: {:.10f}'.format(loss))
  writer.print_line('# BPC: {:.10f}'.format(loss / math.log(2)))
  writer.print_line('# Perplexity: {:.10f}'.format(math.pow(np.e,
                                                            loss)))
  writer.print_line('# Accuracy: {:.10f}'.format(accuracy))
  writer.print_line('# Top-{}: {:.10f}'.format(TOP_K,
                                               top_k_accuracy))
  writer.print_line('# Recall: {:.10f}'.format(recall))
  writer.print_line('# Precision: {:.10f}'.format(precision))
  f_1 = (2 * precision * recall) / (precision + recall)
  writer.print_line('# F1 Score: {:.10f}'.format(f_1))
  if not EVALUATE:
    writer.print_line('# Overfitting Metric: {:.10f}'.format(loss / train_loss))
  writer.print_line('# Time: {}'.format(time.asctime()))
     
  sess.close()
  
  if not EVALUATE:
    writer.print_line('Avg training time {:.4f}'.format(np.mean(metrics['time_per_epoch'])))
  if RUN_PREDICTIONS:
    writer.print_line('Prediction time {:.4f}'.format(time.time() - start_time))
  writer.print_line(metrics)
  writer.print_line(min(metrics['test_loss']))
  
  writer.close_file()

if __name__ == '__main__':
  run(args)