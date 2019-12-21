import numpy as np
import tensorflow as tf

def _create_mask(input_sequence_size,
                 unidirectional = True,
                 out_shape = None):
  '''
  creates the bias for a Transformer network
  if the Transformer network is bidirectional, then the bias is a zeros vector
  if the network is unidirectional, the bias is a mixture of 0 and -1e9, to mask future variables that model would not have access to, at that point, in practice 
  '''
  if unidirectional:
    attn_mask = tf.ones([input_sequence_size, input_sequence_size])
    try:
      mask_u = tf.linalg.band_part(attn_mask, 0, -1)
      mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
    except:
      mask_u = tf.matrix_band_part(attn_mask, 0, -1)
      mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
    ret = (mask_u - mask_dia)  * -1e9
  else:
    ret = tf.zeros([input_sequence_size, input_sequence_size])
  return tf.reshape(ret,
                    [1, 1, input_sequence_size, input_sequence_size])
  
if __name__ == '__main__':
  print(develop_bias(10, 10))