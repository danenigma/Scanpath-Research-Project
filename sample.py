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
def decode_path(vocab, stats, sampled_ids):
	sampled_scanpath = []
	start = 0.0
	for scan_index in sampled_ids:
		if scan_index == 3:
			break
		if scan_index != 2:
			fixation = decode([scan_index], vocab, stats)
			end      = start + fixation[0][2]
			sampled_scanpath.append([fixation[0][0],fixation[0][1], start, end])
			#print([fixation[0][0],fixation[0][1], start, end], fixation[0][2])		
			start   += fixation[0][2] 
	return sampled_scanpath
	
def get_scanpath(vocab, stats, encoder, decoder, image_name):

	image = load_image(image_name)
	image = torch.from_numpy(np.array(image)).transpose(1, 2).transpose(0, 1).float()
	image_tensor = to_var(image.unsqueeze_(0), volatile=True)


	# Generate caption from image
	feature = encoder(image_tensor)
	sampled_ids = decoder.sample(feature)
	sampled_ids = sampled_ids.cpu().data.numpy()
	#print(sampled_ids)
	# Decode word_ids to words
	sampled_scanpath = []
	start = 0.0
	normal_pred = []
	for scan_index in sampled_ids:
		if scan_index == 3:
			break
		if scan_index != 2:
			fixation = decode([scan_index], vocab, stats)
			normal_pred.append(fixation[0])
			end      = start + fixation[0][2]
			sampled_scanpath.append([fixation[0][0],fixation[0][1], start, end])
			#print([fixation[0][0],fixation[0][1], start, end], fixation[0][2])		
			start   += fixation[0][2] 

	return np.array(sampled_scanpath), np.array(normal_pred)
def main(args):

	vocab  = np.load(
			 os.path.join(
			 args.data_dir,
			 '{}-vocab.npy'.format('MIT1003')))

	stats  = np.load(
			 os.path.join(
			 args.data_dir,
			 '{}-stat.npy'.format('MIT1003')), encoding='latin1')
			 
	labels = np.load(
			 os.path.join(
			 args.data_dir,
			 '{}-labels.npy'.format('MIT1003')),
			  encoding='latin1')

	vocab_size  = vocab.reshape(1,-1).shape[1]	
	# Build Models
	encoder = EncoderCNN(args.embed_size)
	encoder.eval()  # evaluation mode (BN uses moving mean/variance)
	decoder = DecoderRNN(args.embed_size, args.hidden_size, 
			             vocab_size, args.num_layers)


	# Load the trained model parameters
	encoder.load_state_dict(torch.load(args.encoder_path))
	decoder.load_state_dict(torch.load(args.decoder_path))
	# If use gpu
	if torch.cuda.is_available():
		encoder.cuda()
		decoder.cuda()

	encoder.eval()  # evaluation mode (BN uses moving mean/variance)
	decoder.eval()  # evaluation mode (BN uses moving mean/variance)
	
	# Prepare Image
	scanpaths = []
	preds = []
	for i, img_name in enumerate(os.listdir('data/FixaTons/MIT1003/STIMULI')):
		full_name = os.path.join('data/FixaTons/MIT1003/STIMULI', img_name)
		target = decode_path(vocab, stats, labels[i][0]+1)
		
		scan   = get_scanpath(vocab, stats, encoder, decoder, full_name)
		preds.append(scan[1])
		scanpaths.append([scan[0], target])
		print(i, full_name)
		
	np.save('scanpaths.npy', np.array(scanpaths))
	np.save('prediction.npy', np.array(preds))
			
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str,
                        help='input image for generating caption')
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
    args = parser.parse_args()
    main(args)
