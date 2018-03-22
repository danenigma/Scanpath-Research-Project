import os
import numpy as np # linear algebra
import torch
import pandas as pd
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch.utils.data.sampler as sampler

from models import *
#from utils import *


class ScanpathDataset(data.Dataset):
    """ scanpath dataset """

    def __init__(self, name, data_directory='data', test=False):
        super(ScanpathDataset, self).__init__()
        self.name = name
        self.data = np.load(os.path.join(data_directory, '{}-feats.npy'.format(self.name)), encoding='latin1')
        if test:
            self.labels = None
        else:
            self.labels = np.load(os.path.join(data_directory, '{}-labels.npy'.format(self.name)), encoding='latin1')
        self.vocab = np.load(os.path.join(data_directory, '{}-vocab.npy'.format(self.name)))
        
    def __getitem__(self, index):
        """
        returns one Utterance (#frames, 40)
        """

        data_torch  = torch.from_numpy(self.data[index]).transpose(2,1).transpose(1,0)
        
        dict_ = {
                'data' : data_torch[:3,:,:],
                }
        
        if self.labels is not None:
            dict_['target'] = torch.LongTensor(self.labels[index][0].tolist())            
        return dict_

    def __len__(self):
        """length of dataset"""
        return self.data.shape[0]


if __name__=='__main__':
    
	batch_size  = 1
	data_directory = 'data'
	val_ds   = ScanpathDataset(name='MIT1003', test=False)

	val_loader = data.DataLoader(
		                         val_ds, batch_size = batch_size,
		                         sampler = RandomSampler(val_ds))

	encoder = EncoderCNN(40)
	#decoder = DecoderRNN(400, 10, 
	#		         len(vocab), 1)

	print(encoder)

	if torch.cuda.is_available():
		encoder.cuda()
		decoder.cuda()

	for batch_index, batch_dict in enumerate(val_loader):
		
		X = Variable(batch_dict['data'])
		Y = Variable(batch_dict['target'], requires_grad = False)
		
		print('X: ', X.shape, 'Y: ', Y.shape)
		if batch_index == 0: break
		
