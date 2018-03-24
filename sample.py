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
    
def main(args):

	vocab  = np.load(
			 os.path.join(
			 args.data_dir,
			 '{}-vocab.npy'.format('MIT1003')))

	stats  = np.load(
			 os.path.join(
			 args.data_dir,
			 '{}-stat.npy'.format('MIT1003')), encoding='latin1')
			 
	vocab_size  = vocab.reshape(1,-1).shape[1]	
	# Build Models
	encoder = EncoderCNN(args.embed_size)
	encoder.eval()  # evaluation mode (BN uses moving mean/variance)
	decoder = DecoderRNN(args.embed_size, args.hidden_size, 
			             vocab_size, args.num_layers)


	# Load the trained model parameters
	encoder.load_state_dict(torch.load(args.encoder_path))
	decoder.load_state_dict(torch.load(args.decoder_path))

	# Prepare Image

	image = load_image(args.image)
	image = torch.from_numpy(np.array(image)).transpose(1, 2).transpose(0, 1).float()
	image_tensor = to_var(image.unsqueeze_(0), volatile=True)

	# If use gpu
	if torch.cuda.is_available():
		encoder.cuda()
		decoder.cuda()

	# Generate caption from image
	feature = encoder(image_tensor)
	sampled_ids = decoder.sample(feature)
	sampled_ids = sampled_ids.cpu().data.numpy()
	print(sampled_ids)

	# Decode word_ids to words
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

	print(np.array(sampled_scanpath).T)
	np.save('png/' + args.image + '.npy', np.array(sampled_scanpath))
	image = Image.open(args.image)
	plt.imshow(np.asarray(image))
	plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
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
