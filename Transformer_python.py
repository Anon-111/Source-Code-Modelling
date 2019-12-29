import os
import time
import numpy as np
import tensorflow as tf

def build_and_run_model(model,
                        load_model = False):
  task = 'code_modelling'
  model_name = model

  import sys
  sys.path.append('Utils/')
  if model == 'evolved_transformer':
    from Evolved_Transformer import Evolved_Transformer as model, argument
  elif model == 'transformer':
    from Transformer import Transformer as model, argument
  elif model == 'transformer_xl':
    from Transformer_XL import Transformer_XL as model, argument
  elif model == 'recurrent_trasformer':
    from Recurrent_Transformer import Recurrent_Transformer as model, argument
  elif model == 'extended_recurrent_transformer':
    from Extended_Recurrent_Transformer import Extended_Recurrent_Transformer as model, argument
  elif model == 'gru':
    from RNN import RNN as model, argument
  elif model == 'lstm':
    from RNN import RNN as model, argument
  else:
    print('Invalid Model')
    sys.exit()

  sys.path.append('Python_Code/')
  from feeder import PythonFeeder

  from optimize import warmup_learning_rate

  def prepare_data(array):
    for i in range(len(array[0])):
      if np.array_equal(array[:,i],
                        np.zeros_like(array[:,i])):
        break
    return array[:,:i]

  arg = argument()
  
  arg.dropout_type = 'vanilla'
  arg.ffd = 'transformer_ffd'
  arg.pos = 'timing'
  
  arg.act_loss_weight = 0.01
  arg.decoder_layers = 6
  arg.encoder_layers = 6
  arg.head_size = 64
  arg.hidden_dim = 256
  arg.filter_size = 1024
  arg.input_vocab_size = 1001
  arg.mem_len = 256
  arg.N = 200
  arg.num_heads = 8
  arg.num_branches = 8
  arg.rnn_encoder_layers = 2
  arg.target_vocab_size = 1001
  
  arg.classification = False
  arg.mask_loss = True
  arg.relative_attention = False
  arg.unidirectional = True
  arg.unidirectional_decoder = True
  arg.unidirectional_encoder = True
  arg.use_act = True
  arg.use_attention = True
  arg.use_decoder = False
  arg.use_mos = False
  arg.use_prev = False
  arg.use_relu = True

  arg.hidden_size = arg.hidden_dim
  arg.layers = arg.encoder_layers
  arg.vocab_size = arg.input_vocab_size

  if model_name == 'gru' or model_name == 'lstm':
    if model_name == 'gru':
      arg.cell = 'gru'
    else:
      arg.cell = 'lstm'
      
    arg.hidden_dim = 256
    arg.layers = 2
      
  if model_name == 'seq2seq':
    arg.stop_feature = 'exp'
    
    arg.use_attention = True
  
  print('loading model')
  model = model(arg)

  lr = warmup_learning_rate(arg.hidden_size ** -0.5,
                            warmup = 10000)

  print(model.name)
  print('Hidden size: {}'.format(arg.hidden_size))
  if model.name == 'RNN' or model.name == 'Seq2Seq':
    print('Cell type: {}'.format(arg.cell))
    print('Layers: {}'.format(arg.layers))
    if model.name == 'RNN':
      print('Unidirectional')
    else:
      print('Bidirectional')
  else:
    print('Layers: {}'.format(arg.encoder_layers))
    if arg.unidirectional_encoder:
      print('Unidirectional')
    else:
      print('Bidirectional')
  print('Code modelling')
  
  if not os.path.exists(model.name):
    os.mkdir(model.name)
  if not os.path.exists(os.path.join(model.name,
                                     task)):
    os.mkdir(os.path.join(model.name,
                          task))

  
  feeder = PythonFeeder(os.listdir('Python_Code/libraries/tokenized-codes/tokenized-codes')[:10000],
                        'Python_Code/libraries/tokenized-codes/tokenized-codes',
                        32,
                        500,
                        length = 1)
  test_feeder = PythonFeeder(os.listdir('Python_Code/libraries/tokenized-codes/tokenized-codes')[10000:],
                             'Python_Code/libraries/tokenized-codes/tokenized-codes',
                             32,
                             500,
                             length = 1)
  
  sess = tf.Session()
  saver = tf.train.Saver()
  if load_model:
    restored_model = os.listdir(os.path.join(model.name,
                                             'code_modelling',
                                             model.name))[2]
    saver = tf.train.import_meta_graph(os.path.join('Results',
                                                    '2-Layer',
                                                    model.name,
                                                    restored_model))
    saver.restore(sess,tf.train.latest_checkpoint(os.path.join('Results',
                                                               '2-Layer',
                                                               model.name)))
    sess.run(tf.global_variables_initializer())
    if restored_model[12] == '_':
      print('Restored model at epoch {}'.format(int(restored_model[11])))
      current_epoch = int(restored_model[11])
    elif restored_model[13] == '_':
      print('Restored model at epoch {}'.format(int(restored_model[11:13])))
      current_epoch = int(restored_model[11:13])
    elif restored_model[14] == '_':
      print('Restored model at epoch {}'.format(int(restored_model[11:14])))
      current_epoch = int(restored_model[11:14])
  else:
    sess.run(tf.global_variables_initializer())
    current_epoch = 0

  max_epochs = 25
  # adjust epochs here
  for epoch in range(current_epoch,
                     max_epochs):
    loss_array = []
    acc_array = []
    for iteration in range(len(feeder)):
      input_data = feeder.get_next_batch()
      memory = np.zeros([arg.encoder_layers, 32, 0, arg.hidden_size])
      for sequence in range(input_data.shape[1]):
        trainX = prepare_data(input_data[:,sequence])
        trainX, trainY = trainX[:,:-1], trainX[:,1:]
        feed_dict = {model.inputs: trainX,
                     model.targets: trainY,
                     model.training: True,
                     model.keep_prob: 0.9,
                     model.learning_rate: lr()}
        feed_dict[model.loss_mask] = np.where(trainY == 0,
                                              0.0,
                                              1.0)
        if model.name == 'Seq2Seq':
          from Seq2Seq import stop_feature
          feed_dict[model.input_stop_feature] = stop_feature(np.shape(trainX)[0],
                                                             np.shape(trainX)[1],
                                                             arg)
          feed_dict[model.target_stop_feature] = stop_feature(np.shape(trainY)[0],
                                                              np.shape(trainY)[1],
                                                              arg)
        if model.name == 'Transformer-XL' or model.name == 'Experimental-Transformer-XL':
          feed_dict[model.memory] = memory
        _, loss, accuracy = sess.run([model.train_op,
                                      model.cost,
                                      model.accuracy],
                                     feed_dict = feed_dict)
        
        if model.name == 'Transformer-XL' or model.name == 'Experimental-Transformer-XL':
          memory = sess.run(model.new_mems,
                            feed_dict = feed_dict)
        loss_array.append(loss)
        acc_array.append(accuracy)
    test_loss_array = []
    test_acc_array = []
    for iteration in range(len(test_feeder)):
      input_data = test_feeder.get_next_batch()
      memory = np.zeros([arg.encoder_layers, 32, 0, arg.hidden_size])
      for sequence in range(input_data.shape[1]):
        testX = prepare_data(input_data[:,sequence])
        testX, testY = testX[:,:-1], testX[:,1:]
        feed_dict = {model.inputs: testX,
                     model.targets: testY,
                     model.training: False,
                     model.keep_prob: 1.0}
        feed_dict[model.loss_mask] = np.where(testY == 0,
                                              0.0,
                                            1.0)
        if model.name == 'Seq2Seq':
          from Seq2Seq import stop_feature
          feed_dict[model.input_stop_feature] = stop_feature(np.shape(testX)[0],
                                                             np.shape(testX)[1],
                                                             arg)
          feed_dict[model.target_stop_feature] = stop_feature(np.shape(testY)[0],
                                                              np.shape(testY)[1],
                                                              arg)
        if model.name == 'Transformer-XL' or model.name == 'Experimental-Transformer-XL':
          feed_dict[model.memory] = memory
        loss, accuracy, correct_prediction, logits = sess.run([model.cost,
                                                       model.accuracy,
                                                       model.correct_prediction,
                                                       model.logits],
                                                      feed_dict = feed_dict)
        if model.name == 'Transformer-XL' or model.name == 'Experimental-Transformer-XL':
          memory = sess.run(model.new_mems,
                            feed_dict = feed_dict)
        test_loss_array.append(loss)
        test_acc_array.append(accuracy)
    print('===============================================')
    testX = np.array([[291, 5, 941, 736, 37, 41, 291,  
                       5, 26, 587, 26, 992, 938, 950,
                       263, 945, 69, 966, 942, 884, 41,  
                       38, 935, 47]])
    feed_dict = {model.inputs: testX,
                 model.training: False,
                 model.keep_prob: 1.0}
    if model.name == 'Transformer-XL' or model.name == 'Experimental-Transformer-XL':
      feed_dict[model.memory] = np.zeros([arg.encoder_layers, 1, 0, arg.hidden_size])
    predict = sess.run(model.predict,
                       feed_dict = feed_dict)
    print(testX[:,1:])
    print(predict)
    print('===============================================')
    print(time.asctime())
    print('Epoch {}, Training Loss {:.4f}, Training Accuracy {:.4f}'.format(epoch + 1,
                                                                            np.mean(loss_array),
                                                                            np.mean(acc_array)))
    print('Test Loss {:.4f}, Test Accuracy {:.4f}'.format(np.mean(test_loss_array),
                                                          np.mean(test_acc_array)))
    print('===============================================')
    print('')
    
    for file in os.listdir(os.path.join(model.name,
                                        task)):
      os.remove(os.path.join(model.name,
                             task,
                             file))
    saver.save(sess,
               os.path.join(model.name,
                            task,
                            'model_epoch{}_of_{}'.format(epoch + 1,
                                                         max_epochs)))
  sess.close()
  
def run(task,
        load_model = False):
  inputs = ['',
            load_model]
  
  if task == 1:
    inputs[0] = 'transformer'
    
  if task == 2:
    inputs[0] = 'evolved_transformer'
    
  if task == 3:
    inputs[0] = 'transformer_xl'
          
  if task == 4:
    inputs[0] = 'recurrent_trasformer'
      
  if task == 5:
    inputs[0] = 'extended_recurrent_transformer'
    
  if task == 6:
    inputs[0] = 'gru'
      
  if task == 7:
    inputs[0] = 'lstm'
      
  build_and_run_model(inputs[0],
                      inputs[1])
                      
run(1)
