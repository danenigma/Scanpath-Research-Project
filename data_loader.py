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
from torchvision import transforms
from PIL import Image
import pandas as pd
from utils import *
def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
		
	return Variable(x, volatile=volatile)

class ScanpathDataset(data.Dataset):
	""" scanpath dataset """

	def __init__(self,data, labels, vocab):
		super(ScanpathDataset, self).__init__()
	
		self.data   = data
		self.labels = labels
		self.vocab  = vocab
		print('data size : ', self.data.shape)

	def __getitem__(self, index):
		"""
		returns 
		"""
		subj = 0#np.random.randint(15)

		data_torch  = torch.from_numpy(self.data[index]).transpose(2,1).transpose(1,0).float()
		image    = data_torch[:3,:,:]
		saliency = data_torch[3:,:,:]
		target   = torch.LongTensor(self.labels[index][subj].tolist()) + 1           
		return image, saliency, target

	def __len__(self):
		"""length of dataset"""
		return self.data.shape[0]

class ScanpathDatasetWithTable(data.Dataset):
	""" scanpath dataset """

	def __init__(self,data, labels, table, vocab):
		super(ScanpathDatasetWithTable, self).__init__()
	
		self.data   = data
		self.labels = labels
		self.vocab  = vocab
		self.table  = table
		
	def __getitem__(self, index):
		"""
		returns 
		"""
		img_index   = self.table[index]
		data_torch  = torch.from_numpy(self.data[img_index]).transpose(2,1).transpose(1,0).float()
		image    = data_torch[:3,:,:]
		saliency = data_torch[3:,:,:]
		target   = torch.LongTensor(self.labels[index].tolist())           
		return image, saliency, target

	def __len__(self):
		"""length of dataset"""
		return self.table.shape[0]
class ScanpathDatasetWithPandas(data.Dataset):
	""" scanpath dataset """
	def __init__(self, data_dir='data', name = 'train', transform=None):
		self.table     = pd.read_pickle(os.path.join(data_dir,'{}_table.pkl'.format(name)))
		self.data_dir  = data_dir
		self.transform = transform

	def __len__(self):
		return len(self.table)
		
	def __getitem__(self, idx):
		img_name = self.table.iloc[idx, 0]
		image  = Image.open(img_name).convert('RGB')
		if self.transform:
			image = self.transform(image)
		target   = torch.from_numpy(np.array(get_scanpath(self.table.iloc[idx, 1])))

		return image, target
def pandas_collate_fn(data):
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
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images, scanpaths = zip(*data)

	# Merge images (from tuple of 3D tensor to 4D tensor).
	images = torch.stack(images, 0)

	# Merge captions (from tuple of 1D tensor to 2D tensor).
	lengths = [len(scan) for scan in scanpaths]
	targets = torch.zeros(len(scanpaths), max(lengths)).long()
	for i, scan in enumerate(scanpaths):
		end = lengths[i]
		targets[i, :end] = scan[:end]        
	return images, targets, lengths
		
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
    
	batch_size  = 4
	data_directory = 'data'
	img_size    = 224 

	train_trans = transforms.Compose([
    	transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	val_ds     = ScanpathDatasetWithPandas(data_dir='data', 
										   name = 'train', 
										   transform=train_trans)
	val_loader = data.DataLoader(
		                         val_ds, batch_size = batch_size,
		                         shuffle = True,
		             			 collate_fn=pandas_collate_fn	
		                         )
	
	for batch_index, (images, targets, lengths) in enumerate(val_loader):
	
		print(images.shape, targets, lengths)


	'''
	name = 'MIT1003'
	train_data = np.load(os.path.join(data_directory, '{}-feats.npy'.format(name)), encoding='latin1')
	labels = np.load(os.path.join(data_directory, '{}-labels.npy'.format(name)), encoding='latin1')
	vocab  = np.load(os.path.join(data_directory, '{}-vocab.npy'.format(name)))

	val_ds   = ScanpathDataset(train_data, labels, vocab)
	val_loader = data.DataLoader(
		                         val_ds, batch_size = batch_size,
		                         sampler = RandomSampler(val_ds),
		                         collate_fn = collate_fn)

	encoder = EncoderCNN(256)
	
	print(encoder)
	vocab_size = vocab.reshape(1,-1).shape[1]
	decoder    = DecoderRNN(256, 512, vocab_size, 1)
	
	print(decoder)
	
	if torch.cuda.is_available():
		encoder.cuda()
		decoder.cuda()
	
	for batch_index, (images, targets, saliencies, lengths) in enumerate(val_loader):
		

		images = to_var(images, volatile=True)
		scanpaths = to_var(targets)
		scanpaths_packed = pack_padded_sequence(scanpaths, lengths, batch_first=True)[0]

		features = encoder(images)
		
		outputs  = decoder(features, scanpaths, lengths)
		
		print('X: ', images.shape, 'Y: ', scanpaths.shape, 'out: ', outputs.shape)
		
		if batch_index == 0: break
	'''
