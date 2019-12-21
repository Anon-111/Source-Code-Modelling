import os
import sys
import numpy as np

try:
	import sentencepiece as spm
	sp = spm.SentencePieceProcessor()
	sp.load('bpe.model')
	
except:
	sp = False
	
def file_to_array(file,
				  file_location,
				  max_size = 1000,
				  length = 10):
	file = os.path.join(file_location,
						file)
	file = open(file,
				'r')
	code = file.readlines()
	if len(code) > 1000:
		#if len(code) > 10000:
		#	print(len(code))
		file.close()
		return False
	file.close()
	
	if len(code) < length:
		return False
	
	array = []
	maxlen = -1
	for array_line in range(len(code) - length + 1):
		for code_line in range(length):
			if code_line == 0:
				array.append(eval(code[array_line + code_line][:-1])) #[1:])
			else:
				array[-1] += eval(code[array_line + code_line][:-1]) #[1:]
		if len(array[-1]) > maxlen:
			maxlen = len(array[-1])
			if maxlen > max_size:
				return False
	
	np_array = np.zeros([len(array),
						 maxlen])
	for l in range(len(array)):
		np_array[l][0:len(array[l])] = array[l]
	
	return np_array	

def file_to_batch(files,
				  file_location,
				  max_size = 1000,
				  length = 10):
	failed_to_convert = 0
	batch_array = []
	max_rows = -1
	max_cols = -1
	for file in files:
		x = file_to_array(file,
						  file_location,
						  max_size = max_size,
						  length = length)
		if type(x) != bool:
			batch_array.append(x)
			if x.shape[0] > max_rows:
				max_rows = x.shape[0]
			if x.shape[1] > max_cols:
				max_cols = x.shape[1]
		else:
			failed_to_convert += 1
	if max_rows == max_cols == -1:
		assert failed_to_convert == len(files)
		return [], failed_to_convert
	np_batch_array = np.zeros([len(batch_array),
							   max_rows,
							   max_cols])
	for c in range(len(batch_array)):
		np_batch_array[c,0:batch_array[c].shape[0],0:batch_array[c].shape[1]] = batch_array[c]
		
	return np_batch_array, failed_to_convert

class PythonFeeder():
	def __init__(self, files,
				 file_location,
				 batch_size,
				 max_size,
				 length = 10,
				 test_do_not_keep = True):
		self.files = files
		self.file_location = file_location
		self.batch_size = batch_size
		self.max_size = max_size
		self.length = length
		
		self.counter = 0
		
		self.test_do_not_keep = test_do_not_keep
		
	def __len__(self):
		return len(self.files) // self.batch_size + 1
		
	def get_next_batch(self):
		'''
		the output is code array of shape [batch_size, xl-sequence_size, sequence_size]
		i.e. given array of shape [b, x, s], use following code
		
		for i in range(x):
			data = array[:,i,:]
		
		if xl, use memory across x, otherwise dont
		
		'''
		if self.counter + self.batch_size > len(self.files):
			hold = self.files[self.counter:]
			np.random.shuffle(self.files[:self.counter])
			self.files = hold + self.files
			self.counter = 0
		np_array, fail_to_convert = file_to_batch(self.files[self.counter:self.counter + self.batch_size],
												  self.file_location,
												  self.max_size,
												  self.length)
		
		self.counter += self.batch_size
		while fail_to_convert != 0:
			if self.counter + fail_to_convert > len(self.files):
				hold = self.files[self.counter:]
				np.random.shuffle(self.files[:self.counter])
				self.files = hold + self.files
				self.counter = 0
			np_array_two, fail_to_convert_two = file_to_batch(self.files[self.counter:self.counter + fail_to_convert],
															  self.file_location,
															  self.max_size,
															  self.length)
			
			if fail_to_convert != fail_to_convert_two:
				if self.test_do_not_keep:
					if type(np_array) == list and type(np_array_two) != list:
						np_array = np_array_two
					else:
						np_array = self.combine_array(np_array,
													  np_array_two)
				else:
					if type(np_array) == list or type(np_array_two) == list:
						print(self.counter)
						print(np_array)
						print(type(np_array_two))
						exit()
					np_array = self.combine_array(np_array,
												  np_array_two)
				
			self.counter += fail_to_convert
			fail_to_convert = fail_to_convert_two
		return np_array
		
	def combine_array(self, array_one,
					  array_two):
		max_rows = max(array_one.shape[1],
					   array_two.shape[1])
		max_cols = max(array_one.shape[2],
					   array_two.shape[2])
		
		np_array = np.zeros([array_one.shape[0] + array_two.shape[0],
							 max_rows,
							 max_cols])
		np_array[0:array_one.shape[0],0:array_one.shape[1],0:array_one.shape[2]] = array_one
		np_array[array_one.shape[0]:,0:array_two.shape[1],0:array_two.shape[2]] = array_two
		return np_array
		
		
	def change_batch_size(self, batch_size):
		self.batch_size = batch_size
		
	def change_max_size(self, max_size):
		self.max_size = max_size
		
	def change_length(self, length):
		self.length = length

if __name__ == '__main__':
	
	import time
	start_time = time.time()
	
	files = os.listdir(os.path.join('libraries',
									'tokenized-codes'))
	
	feeder = PythonFeeder(files = os.listdir(os.path.join('libraries',
														  'tokenized-codes')),
						  file_location = os.path.join('libraries',
													   'tokenized-codes'),
						  batch_size = 32,
						  max_size = 500,
						  length = 1)
	
	print(len(feeder))
	
	for i in range(10):
		np_array = feeder.get_next_batch()
		print(np.shape(np_array))
        
	print(time.time() - start_time)