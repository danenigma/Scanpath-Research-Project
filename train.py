import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from models import EncoderCNN, DecoderRNN 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from data_loader import *

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
    
def main(args):

	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

	train_data = np.load(
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
			 '{}-vocab.npy'.format(args.name)))
	vocab_size  = vocab.reshape(1,-1).shape[1]			 
	scanpath_ds = ScanpathDataset(train_data, labels, vocab)
	data_loader = data.DataLoader(
			                     scanpath_ds, batch_size = args.batch_size,
			                     sampler = RandomSampler(scanpath_ds),
			                     collate_fn = collate_fn)
			                     
	encoder = EncoderCNN(args.embed_size)
	decoder = DecoderRNN(args.embed_size, args.hidden_size, 
			             vocab_size, args.num_layers)

	if torch.cuda.is_available():
		encoder.cuda()
		decoder.cuda()

	# Loss and Optimizer
	criterion = nn.CrossEntropyLoss()
	params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
	optimizer = torch.optim.Adam(params, lr=args.learning_rate)
	total_step = len(data_loader)
	print('total_steps: ', total_step)

	for epoch in range(args.num_epochs):
	
		for i, (images, targets, saliencies, lengths) in enumerate(data_loader):
		
			# Set mini-batch dataset
			images = to_var(images, volatile=True)
			scanpaths = to_var(targets)
			scanpaths_packed = pack_padded_sequence(scanpaths, lengths, batch_first=True)[0]

			# Forward, Backward and Optimize
			decoder.zero_grad()
			encoder.zero_grad()
			features = encoder(images)
			outputs = decoder(features, scanpaths, lengths)

			loss = criterion(outputs, scanpaths_packed)
			loss.backward()
			optimizer.step()

			# Print log info
			if i % args.log_step == 0:
				print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
					  %(epoch, args.num_epochs, i, total_step, 
						loss.data[0], np.exp(loss.data[0]))) 
	
			# Save the models
			if (i+1) % args.save_step == 0:
				torch.save(decoder.state_dict(), 
						   os.path.join(args.model_path, 
							            'decoder-%d-%d.pkl' %(epoch+1, i+1)))
				torch.save(encoder.state_dict(), 
						   os.path.join(args.model_path, 
							            'encoder-%d-%d.pkl' %(epoch+1, i+1)))
			
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' ,
                        help='path for saving trained models')
    parser.add_argument('--data_dir', type=str, default='data/' ,
                        help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=2,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000,
                        help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--name', type=str, default='MIT1003')
    
    args = parser.parse_args()
    main(args)
