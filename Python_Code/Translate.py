import os
import sys
import numpy as np

def id2piece():
  idx2word = {0: '<stop>'}
  vocab = open('bpe.vocab',
               'r').readlines()
  for i in range(len(vocab)):
    idx2word[i] = vocab[i].split()[0]
  return idx2word
    
def vocabulary_of_note():
  vocab = {'as': 5,
           'from': 14,
           '<indent>': 3,
           ':': 39,
           ',': 946,
           '.': 26,
           '(': 37,
           ')': 38,
           '{': 35,
           '}': 36,
           '[': 33,
           ']': 34,
           'if': 6,
           'elif': 7,
           'else': 8,
           'for': 10,
           'while': 9,
           '=': 30}
    
def piece2id(idx2word):
  return {v: k for (k, v) in idx2word.items()}

def translate(line,
              idx2word):
  string = ''
  for token in line:
    if token != 0:
      if idx2word[token] == '<indent>':
        string += ' '
      else:
        string += idx2word[token]
  return string.replace('▁', ' ')
  
class Translate():
  def __init__(self):
    try:
      self.idx2word = id2piece()
      self.word2idx = piece2id(self.idx2word)
    except:
      self.idx2word = None
      self.word2idx = None
    
  def translate(self, data):
    for line in data:
      if sum(line) != 0:
        print(translate(line,
                        self.idx2word))

if __name__ == '__main__':
  def prepare_data(array):
    for i in range(len(array[0])):
      if np.array_equal(array[:,i],
                        np.zeros_like(array[:,i])):
        break
    return array[:,:i]
  
  errors_class = Translate()

  translate_per_batch = False # translate across batch, or translate across sequence

  from feeder import PythonFeeder
  
  feeder = PythonFeeder(files = os.listdir(os.path.join('libraries',
                                                        'tokenized-codes')),
                        file_location = os.path.join('libraries',
                                                     'tokenized-codes'),
                        batch_size = 4,
                        max_size = 500,
                        length = 1)
  
  np_array = feeder.get_next_batch()

  if translate_per_batch:
    np_array = prepare_data(np_array[:,1,:])
  else:
    np_array = prepare_data(np_array[1,:,:])
    
  errors_class.translate(np_array)