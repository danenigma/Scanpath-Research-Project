import torch
import torch.utils.data as data
from PIL import Image
from os import listdir
from os.path import join
import pandas as pd
import torchvision.transforms as transforms
from utils import *

class ScanpathDatasetAttention(data.Dataset):
	""" scanpath dataset """
	def __init__(self, data_dir='data', 
					   name = 'train', 
					   feature_path = 'data/mit1003.pth'):
					   
		self.table      = pd.read_pickle(join(data_dir,'{}_table.pkl'.format(name)))
		self.feat_table = pd.read_pickle(join(data_dir,'feat_table.pkl'))
		#print(self.feat_table)
		self.data_dir  = data_dir
		self.features  = torch.load(feature_path)
		
	def __len__(self):
		return len(self.table)
		
	def __getitem__(self, idx):
	
		img_name = self.table.iloc[idx, 0]
		feat_idx = int(self.feat_table[
					   self.feat_table['image_path']==img_name].index[0])
		#feat_idx = 0
		feature  = self.features[feat_idx].view(512,-1)
		target   = torch.from_numpy(np.array(get_scanpath(self.table.iloc[idx, 1])))

		return feature, target
	def collate_fn(self, data):
		data.sort(key=lambda x: len(x[1]), reverse=True)
		features, captions = zip(*data)

		features = torch.stack(features, 0)
		# batch_size-by-512-196

		lengths = [len(cap) for cap in captions]
		targets = torch.zeros(len(captions), max(lengths)).long()
		for i, cap in enumerate(captions):
		    end = lengths[i]
		    targets[i, :end] = cap[:end]

		return features, targets, lengths
def get_loader(data_dir='data', 
			   name = 'train', 
			   feature_path = 'data/mit1003.pth', 
			   batch_size=1,
               shuffle=False, 
               num_workers=2):

    ds = ScanpathDatasetAttention(data_dir=data_dir, 
					   				 name = name, 
					   				 feature_path = feature_path)
					   

    data_loader = torch.utils.data.DataLoader(dataset=ds,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=ds.collate_fn)
    return data_loader


if __name__=='__main__':

	data_loader =  get_loader(data_dir='data', 
			  				  name = 'train', 
			   				  feature_path = 'data/mit1003.pth', 
			   				  batch_size=2,
               				  shuffle=False, 
               				  num_workers=2)
               				  
	for i, (feats, targets, lengths) in enumerate(data_loader):
		print(feats.shape, targets)
		break
