import os
import sys
import random
import numpy as np

class Python_Feeder():
  # Feeder is the overall class too collect, tokenize present the data
  def __init__(self, batch_size = 128,
               tokenization_scheme = 'character'):
    # self.libraries is the dict that holds the folders where .py codes are, and the number of .py files that are held in each
    # further, references is where each library was taken
    
    self.batch_size = batch_size
    self.tokenization_scheme = tokenization_scheme
    assert self.tokenization_scheme == 'character' or self.tokenization_scheme == '1000-bpe'
    if self.tokenization_scheme == '1000-bpe':
      try:
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join('Source-Code-BPE',
                                  'source-code-bpe.model'))
        self.vocab = self.load_vocab()
      except:
        pass
    
    self.libraries = {'TheAlgorithms/': 473,
                      'geekcomputers/': 227,
                      'algorithms-master/': 360,
                      'python-patterns-master/': 61,
                      'cryptography-master/': 227,
                      'pandas-master/': 869,
                      'mining-master/': 43,
                      'cerberus-master/': 47,
                      'bokeh-master/': 1074,
                      'altair-master/': 242,
                      'matplotlib-master/': 859,
                      'dateutil-master/': 38,
                      'cartridge-master/': 46,
                      'PySimpleGUI-master/': 443,
                      'Tkinter-By-Example-master/': 34,
                      'toga-master/': 498,
                      'Pillow-master/': 275,
                      'scikit-image-master/': 523,
                      'cpython-master/': 1941 - 13,
                      'scikit-learn-master/': 770,
                      'Metrics-master/': 16,
                      'stanfordnlp-master/': 107,
                      'gensim-develop/': 227,
                      'ros_comm-melodic-devel/': 305,
                      'pyspider-master/': 108,
                      'sympy-master/': 1353,
                      'statsmodels-master/': 1047,
                      'scipy-master/': 728,
                      'numpy-master/': 430,
                      'qutip-master/': 180,
                      'pygame-master/': 212,
                      'tensorflow-master/': 2477,
                      'libcloud-trunk/': 717,
                      'blaze-master/': 133}
    self.libraries_reference = {'TheAlgorithms/': 'https://github.com/TheAlgorithms/Python',
                                'geekcomputers/': 'https://github.com/geekcomputers/Python',
                                'algorithms-master/': 'https://github.com/keon/algorithms',
                                'python-patterns-master/': 'https://github.com/faif/python-patterns',
                                'cryptography-master/': 'https://github.com/pyca/cryptography',
                                'pandas-master/': 'https://github.com/pandas-dev/pandas',
                                'mining-master/': 'https://github.com/mining/mining',
                                'cerberus-master/': 'https://github.com/pyeve/cerberus',
                                'bokeh-master/': 'https://github.com/bokeh/bokeh',
                                'altair-master/': 'https://github.com/altair-viz/altair',
                                'matplotlib-master/': 'https://github.com/matplotlib/matplotlib',
                                'dateutil-master/': 'https://github.com/dateutil/dateutil',
                                'cartridge-master/': 'https://github.com/stephenmcd/cartridge',
                                'PySimpleGUI-master/': 'https://github.com/PySimpleGUI/PySimpleGUI',
                                'Tkinter-By-Example-master/': 'https://github.com/Dvlv/Tkinter-By-Example',
                                'toga-master/': 'https://github.com/beeware/toga',
                                'Pillow-master/': 'https://github.com/python-pillow/Pillow',
                                'scikit-image-master/': 'https://github.com/scikit-image/scikit-image',
                                'cpython-master/': 'https://github.com/python/cpython',
                                'scikit-learn-master/': 'https://github.com/scikit-learn/scikit-learn',
                                'Metrics-master/': 'https://github.com/benhamner/Metrics',
                                'stanfordnlp-master/': 'https://github.com/stanfordnlp/stanfordnlp',
                                'gensim-develop/': 'https://github.com/RaRe-Technologies/gensim',
                                'ros_comm-melodic-devel/': 'https://github.com/ros/ros_comm',
                                'pyspider-master/': 'https://github.com/binux/pyspider',
                                'sympy-master/': 'https://github.com/sympy/sympy',
                                'statsmodels-master/': 'https://github.com/statsmodels/statsmodels',
                                'scipy-master/': 'https://github.com/scipy/scipy',
                                'numpy-master/': 'https://github.com/numpy/numpy',
                                'qutip-master/': 'https://github.com/qutip/qutip',
                                'pygame-master/': 'https://github.com/pygame/pygame/',
                                'tensorflow-master/': 'https://github.com/tensorflow/tensorflow',
                                'libcloud-trunk/': 'https://github.com/apache/libcloud',
                                'blaze-master/': 'https://github.com/blaze/blaze'}
    self.python_files = []
    for file in self.libraries_reference.keys():
      for root, dirs, files in os.walk(file):
        for file in files:
          if file.endswith('.py'):
            self.python_files.append(os.path.join(root,
                                                  file))
    #self.explore()
    try:
      data = np.load('saved-data/{}.npy'.format(self.tokenization_scheme))
    except:
      data = np.load('{}.npy'.format(self.tokenization_scheme))
    
    sequence_size = len(data) // self.batch_size
    self.seed_txt = [data[self.batch_size * sequence_size:]]
    data = np.reshape(data[:self.batch_size * sequence_size],
                      [self.batch_size, sequence_size])
    self.train_data = data[:,:(-1 * sequence_size // 10 * 2)]
    self.valid_data = data[:,(-1 * sequence_size // 10 * 2):(-1 * sequence_size // 10)]
    self.test_data = data[:,(-1 * sequence_size // 10):]
    
  def __call__(self, task = 'train',
               sequence_size = 100):
    sequence_size += 1
    # this is p(y_{t+1} | y_t, ..., y_1), so an additional time-step is added
    if task == 'train':
      data = self.train_data[:,:sequence_size]
      self.train_data = np.concatenate([self.train_data[:,sequence_size:], data],
                                       axis = 1)
    elif task == 'valid':
      data = self.valid_data[:,:sequence_size]
      self.valid_data = np.concatenate([self.valid_data[:,sequence_size:], data],
                                       axis = 1)
    elif task == 'test':
      data = self.test_data[:,:sequence_size]
      self.test_data = np.concatenate([self.test_data[:,sequence_size:], data],
                                      axis = 1)
    return data
    
  def explore(self):
    print('Number of Python Files: {}'.format(len(self.python_files)))
    # 17077
    tokenized_code = []
    no_of_lines = 0
    for file in self.python_files:
      comments = False
      file = open(file, 'r')
      for line in file:
        line, comments = self.prepare_line(line, comments)
        # knocking out '#' and '''
        # if line is False, then the line has no information
        # if comments is True, then the line has no information
        if type(line) == bool:
          assert not line
        else:
          if not comments:
            tokenized_line = self.translate_to_idx(line)
            tokenized_code.extend(tokenized_line)
            no_of_lines += 1
      tokenized_code += [1]
      # 1 refers to end of line
    print('Number of lines: {}'.format(no_of_lines))
    # 3,895,505
    print('Number of tokens: {}'.format(len(tokenized_code)))
    # 88,993,431 # subword
    # 141,568,827 # characters
    np.save('saved-data/{}.npy'.format(self.tokenization_scheme),
            tokenized_code)
    
  def translate_to_idx(self, line):
    if self.tokenization_scheme == 'character':
      return np.array([ord(c) for c in line if ord(c) < 255])
    elif self.tokenization_scheme == '1000-bpe':
      # 2 refers to \n
      return self.sp.encode_as_ids(line) + [2]
  
  def translate_to_word(self, line):
    if self.tokenization_scheme == 'character':
      return ''.join([chr(c) for c in line if c != 0])
    elif self.tokenization_scheme == '1000-bpe':
      text = ''
      for token in line:
        if token != 8000:
          token = int(token)
          text += self.vocab[token]
      text = text.replace('<indent>', ' ')
      return text
      
  def prepare_line(self, line,
                   comments = False):
    # the first thing is finding the existence of either ''' or """
    # there are 3 variants of triple-italics line of code
    # 1. two triple-italic, e.g. '''this is a single line comment'''
    # 2. single triple-italic e.g. '''the next line will also be a comment
    # 3. comment triple-italic e.g. #'''this is not a triple-comment
    
    # the first case comments out the line, but the comment ends at the end of the line. The next line will be analyzed
    # the second case does not end the comment, and the next line will be ignored
    # the third comments out the ''' using # entirely. This line is commented out, but because of # and not '''. This is not a triple-italic comment. The next line will be analyzed
    # because of the difference in analysis from the next line on, it is crucial that the line is correctly classified
    if '"""' in line or "'''" in line:
      two_triple_italics = False
      # two_triple_italics = False, at the end, means there are two triple-italics, the first variants
      # if two_triple_italics = True, then there is only one triple-italics. Therefore the second variant
      for token in range(len(line)):
        if line[token] == '"' or line[token] == "'":
          if token + 2 < len(line):
            if line[token] == line[token + 1] == line[token + 2]:
              # current seeing if the token at point t is the first italic of a three-italic system
              two_triple_italics = not two_triple_italics
              # if a triple-italic is found, then not two_triple_italics
        elif line[token] == '#' and not two_triple_italics:
          # if # comes before ''', then the third variant. Line is not analyzed, but the next line will be
          return False, comments
      if two_triple_italics:
        # only one ''' was found
        # if the previous line was analyzed, then this is the beginning of a comment block and will not be analyzed until the next ''' is found
        # if the previous line was not analyzed, then the comment block is over at the next line
        return False, not comments
      else:
        # this was the first variant. The next line will be analyzed
        return False, comments
    
    # if there is no ''' in the line, but comments = True, then this is still a comment block and the line is ignored
    if comments:
      return False, comments
    
    # all indentation whitespaces are replaced by <indent>
    # this is because, unlike most other coding languages, the indentation of python code is essential. A change to the indentation changes how the code is read
    # therefore, each whitespace before the coding begins is replaced by an <indent> so the indentation level can be analyzed
    start = False
    # once start = True, the line of code has passed the indentation level. ' ' will no longer be replaced with <indent>
    loc = ''
    for token in line:
      if token != ' ' and token != '#':
        # if the token is anything other then ' ', '#', then it will be added.
        # this is a result of the decision by the authors to minimize the amount of preprocessing of the source code
        start = True
        loc += token
      elif token == '#' and not start:
        # this is a sign that the line of code is # comment, and is therefore ignored entirely
        return False, comments
      elif token == '#' and start:
        # this line of code ends with a comment, but there is a line of code beforehand. 
        # the beforehand line of code is returned and the comment is ignored
        return loc, comments
      elif token == ' ' and not start:
        # if token == ' ' and before the start-of-the-line, then <indent>
        if self.tokenization_scheme == '1000-bpe':
          loc += '<indent>'
          # 1000-bpe contains the hyperparameter <indent>
          # 2 is set to \n
        else:
          loc += ' '
      elif token == ' ' and start:
        # else, it is still whitespace
        loc += ' '
    return loc, comments
    
  def load_vocab(self):
    vocab_file = open(os.path.join('Source-Code-BPE',
                                   'source-code-bpe.VOCAB'),
                      'r')
    vocab_dict = {}
    iteration = 0
    for line in vocab_file:
      line = line.split('\t')
      vocab_dict[iteration] = line[0].replace('▁', ' ')
      iteration += 1
    vocab_dict[1] = 'new file \n'
    vocab_dict[2] = '\n'
    return vocab_dict
    
if __name__ == '__main__':
  feeder = Python_Feeder(tokenization_scheme = 'character')
  
  SEQ_LEN = 100
  
  data = feeder(sequence_size = SEQ_LEN)
  print(np.shape(data))
  data = feeder(task = 'valid',
                sequence_size = SEQ_LEN)
  print(np.shape(data))
  data = feeder(task = 'test',
                sequence_size = SEQ_LEN)
  print(np.shape(data))
  
  print(np.shape(feeder.seed_txt))
  print(feeder.translate_to_word(feeder.seed_txt[0]))