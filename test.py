import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torch.autograd import Variable 
from torchvision import transforms 
from models import EncoderCNN, DecoderRNN
from PIL import Image
import decoder as dec
from data_loader import *
import scipy.io as sio

def decode(scanpath, vocab, stats):
	output = []
	for scan in scanpath:
			x_idx, y_idx, dur_idx = np.where(vocab == scan)		
			x_mean, x_std = stats[0][x_idx[0]], stats[3][x_idx[0]]
			y_mean, y_std = stats[1][y_idx[0]], stats[4][y_idx[0]]
			dur_mean, dur_std = stats[2][dur_idx[0]], stats[5][dur_idx[0]]
		
			x   = np.random.normal(x_mean, x_std/2)
			y   = np.random.normal(y_mean, y_std/2)
			dur = np.random.normal(dur_mean, dur_std/2)
		
			output.append([int(x), int(y), dur])
	
	return np.array(output)
def to_var(x, volatile=False):

    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def load_image(image_path):

    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    
    return image
    
def main(args):

	img_data = np.load(
				 os.path.join(
				 args.data_dir,
				 '{}-feats.npy'.format(args.name))
				 , encoding='latin1')
			 
	labels = np.load(
			 os.path.join(
			 args.data_dir,
			 '{}-labels.npy'.format(args.name)),
			  encoding='latin1')

	vocab  = np.load(
			 os.path.join(
			 args.data_dir,
			 '{}-vocab.npy'.format('MIT1003')))

	stats  = np.load(
			 os.path.join(
			 args.data_dir,
			 '{}-stat.npy'.format('MIT1003')), encoding='latin1')
			 
	vocab_size  = vocab.reshape(1,-1).shape[1]	
	scanpath_ds = ScanpathDataset(img_data, labels, vocab)

	data_loader = data.DataLoader(
					             scanpath_ds, batch_size = 1,
					             shuffle = False,
					             collate_fn = collate_fn)

	# Build Models
	encoder = EncoderCNN(args.embed_size)
	encoder.eval()  # evaluation mode (BN uses moving mean/variance)
	decoder = DecoderRNN(args.embed_size, args.hidden_size, 
			             vocab_size, args.num_layers)


	# Load the trained model parameters
	encoder.load_state_dict(torch.load(args.encoder_path))
	decoder.load_state_dict(torch.load(args.decoder_path))
	print("MODEL LOADING DONE!")
	# If use gpu
	if torch.cuda.is_available():
		encoder.cuda()
		decoder.cuda()
	scanpaths = []
	target_scanpaths = []
	# Prepare Image
	for i, (image, target, saliency, length) in enumerate(data_loader):

		image = to_var(image, volatile=True)
		target_scanpath = target.cpu().numpy()[0][1:-1]
		feature = encoder(image)
		sampled_ids = decoder.sample(feature)
		sampled_ids = sampled_ids.cpu().data.numpy()

		# Decode word_ids to words
		sampled_scanpath = []
		sampled_target = []
		start = 0.0
		
		for index, scan_index in enumerate(target_scanpath):
			if scan_index == 3:
				break
			if scan_index != 2:
				try:
					fixation = decode([scan_index], vocab, stats)
					fixation[0][2]*=1000
					sampled_scanpath.append(fixation[0])
				except:
					#print('error') 
					pass
					
		#target_scanpath = decode(target_scanpath, vocab, stats)
		#print(target_scanpath)
		if i % 100==0:print('*****[{:d}]*****'.format(i))
		sampled_scanpath = np.array(sampled_scanpath)
		#target_scanpaths.append(target_scanpath)
		#print(sampled_scanpath.shape)
		scanpaths.append(sampled_scanpath)
		
	#print(len(scanpaths))	
	scanpaths = np.array(scanpaths)
	
	sio.savemat('target_scanpaths', {'target_scanpaths': scanpaths})
	#sio.savemat('target_scanpaths', {'target_scanpaths': target_scanpaths})
	
		#print(np.array(sampled_scanpath).T)
		#np.save(args.image + '.npy', np.array(sampled_scanpath))
		#image = Image.open(args.image)
		#plt.imshow(np.asarray(image))
		#plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='./models/encoder-15-1.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/decoder-15-1.pkl',
                        help='path for trained decoder')
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    parser.add_argument('--data_dir', type=str, default='data/' ,
                        help='directory for resized images')
    parser.add_argument('--name', type=str, default='MIT1003' ,
                        help='directory for resized images')
                        
    args = parser.parse_args()
    main(args)
