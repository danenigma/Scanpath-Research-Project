import os
import numpy as np # linear algebra
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch.utils.data.sampler as sampler
import matplotlib.pyplot as plt
from models import *
#from utils import *
def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
		
	return Variable(x, volatile=volatile)

class MultiMatchDataset(data.Dataset):
	""" scanpath dataset """

	def __init__(self,data, labels):
		super(MultiMatchDataset, self).__init__()
	
		self.data   = data
		self.labels = labels

		print('data size : ', self.data.shape)

	def __getitem__(self, index):
		"""
		returns 
		"""

		seq_1  = torch.from_numpy(self.data[index][0]).float()
		seq_2  = torch.from_numpy(self.data[index][1]).float()
		target = torch.FloatTensor(self.labels[index])           
		return seq_1, seq_2, target

	def __len__(self):
		"""length of dataset"""
		return self.data.shape[0]


def collate_fn(data):
	"""Creates mini-batch tensors from the list of tuples (image, caption).

	We should build custom collate_fn rather than using default collate_fn, 
	because merging caption (including padding) is not supported in default.
	Args:
		data: list of tuple (image, caption). 
		    - image: torch tensor of shape (3, 256, 256).
		    - caption: torch tensor of shape (?); variable length.
	Returns:
		images: torch tensor of shape (batch_size, 3, 256, 256).
		targets: torch tensor of shape (batch_size, padded_length).
		lengths: list; valid length for each padded caption.
	"""
	# Sort a data list by caption length (descending order).
	data.sort(key=lambda x: len(x[2]), reverse=True)
	images, saliencies, captions = zip(*data)

	# Merge images (from tuple of 3D tensor to 4D tensor).
	images = torch.stack(images, 0)
	saliencies = torch.stack(saliencies, 0)

	# Merge captions (from tuple of 1D tensor to 2D tensor).
	lengths = [len(cap) for cap in captions]
	targets = torch.zeros(len(captions), max(lengths)).long()
	for i, cap in enumerate(captions):
		end = lengths[i]
		targets[i, :end] = cap[:end]        
	return images, targets, saliencies, lengths

if __name__=='__main__':
    
	batch_size  = 1
	train_data = np.load('seq_pairs.npy')
	labels = np.load('multi_matchs.npy')

	print(train_data.shape, labels.shape)
	val_ds     = MultiMatchDataset(train_data, labels)
	val_loader = data.DataLoader(
		                 val_ds, batch_size = batch_size,
		                 sampler = RandomSampler(val_ds)
		                 )
	model = MultiMatchLoss()
	print(model)
	             
	for batch_index, (seq_1, seq_2, target) in enumerate(val_loader):
		print(seq_1.shape, seq_2.shape)
		seq_1 = to_var(seq_1)
		seq_2 = to_var(seq_2)
		
		seq_1_lens = [seq_1.shape[0]]
		seq_2_lens = [seq_2.shape[0]]
		#seq_1_packed = pack_padded_sequence(seq_1, seq_1_lens, batch_first=True)[0]
		#seq_2_packed = pack_padded_sequence(seq_2, seq_2_lens, batch_first=True)[0]
		
		out = model(seq_1, seq_2, seq_1_lens, seq_2_lens)
		print(out)
		break
		
