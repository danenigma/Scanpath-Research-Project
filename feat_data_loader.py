import torch
import torch.utils.data as data
from PIL import Image
from os import listdir
from os.path import join
import pandas as pd
import torchvision.transforms as transforms

class ImageExtractDataset(data.Dataset):

	def __init__(self, data_dir = 'data', transform=None):
		self._transform = transform
		self.table      = pd.read_pickle(join(data_dir,'feat_table.pkl'))

	def __getitem__(self, index):

		image = self.table.iloc[index, 0] 
		image = Image.open(image).convert('RGB')
		image = image.resize([224, 224], Image.LANCZOS)
		if self._transform is not None:
			image = self._transform(image)
			
		return image

	def __len__(self):
		return len(self.table)

def get_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    return transform


def get_loader(data_dir, transform, batch_size, shuffle, num_workers):

    dataset = ImageExtractDataset(data_dir = data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader( dataset=dataset, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers)

    return data_loader



if __name__=='__main__':
	name = 'feat'
	#table     = pd.read_pickle(join(data_dir,'{}_table.pkl'.format(name)))
	
	data_dir  = 'data'
	transform = get_transform()
	batch_size = 10
	shuffle = True
	num_workers = 0 
	loader    = get_loader(data_dir, 
						transform, 
						batch_size, 
						shuffle, 
						num_workers)

	for i, images in enumerate(loader):
		print(images.shape)
		break
