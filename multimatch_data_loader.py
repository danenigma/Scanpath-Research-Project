import os
import numpy as np # linear algebra
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch.utils.data.sampler as sampler
import matplotlib.pyplot as plt
import scipy.io as sio
from models import *
#from utils import *

def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
		
	return Variable(x, volatile=volatile)

class MultiMatchDataset(data.Dataset):
	""" scanpath dataset """

	def __init__(self, data, table, labels):
		super(MultiMatchDataset, self).__init__()
	
		self.data   = data
		self.labels = labels
		self.table  = table

		print('data size : ', self.data.shape)

	def __getitem__(self, index):
		"""
		returns 
		"""
		
		seq_1_idx, seq_2_idx  = self.table[index][0], self.table[index][1]
		seq1_np = self.data[seq_1_idx-1][0]#because index starts at 1 in matlab
		seq2_np = self.data[seq_2_idx-1][0]
		seq_1  = torch.from_numpy(seq1_np).float()
		seq_2  = torch.from_numpy(seq2_np).float()
		
		target = torch.FloatTensor(self.labels[index])           
		return seq_1, seq_2, target

	def __len__(self):
		"""length of dataset"""
		return self.table.shape[0]


if __name__=='__main__':
    
	batch_size  = 1
	train_data = sio.loadmat('./data/scanpaths.mat')['target_scanpaths'].T
	labels     = sio.loadmat('./data/multimatch_target.mat')['output']
	table      = sio.loadmat('./data/multimatch_input.mat')['input']
	

	print(train_data.shape, labels.shape)
	train_ds     = MultiMatchDataset(train_data,table, labels)
	train_loader = data.DataLoader(
			             train_ds, batch_size = batch_size,
			             sampler = RandomSampler(train_ds)
			             )
	
	model = MultiMatchLoss()
	print(model)
	criterion = nn.MSELoss()
	total_loss = 0
	  
	for batch_index, (seq_1, seq_2, target) in enumerate(train_loader):
		
		
		seq_1  = to_var(seq_1)
		seq_2  = to_var(seq_2)
		target = to_var(target)
	
		out = model(seq_1, seq_2)
		loss = criterion(out, target)
		
		print(out, loss)
	
		break
		
