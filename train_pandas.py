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
def validate(encoder, decoder, data_loader, criterion):

	val_loss = 0
	encoder.eval()
	decoder.eval()

	for i, (images, targets, lengths) in enumerate(data_loader):
		bsz = images.shape[0]
		# Set mini-batch dataset
		images = to_var(images, volatile=True)
		scanpaths = to_var(targets)
		scanpaths_packed = pack_padded_sequence(scanpaths, lengths, batch_first=True)[0]

		features = to_var(torch.zeros(bsz,256))#encoder(images)
		print('feats: ', features.shape)
		outputs  = decoder(features, scanpaths, lengths)

		loss = criterion(outputs, scanpaths_packed)
		val_loss = loss.data.sum()
		if i%100==0:
			print('[index {}]'.format(i))
	return val_loss
    
def main(args):

	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

	data_directory = 'data'
	img_size    = 224 

	trans = transforms.Compose([
    	transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	val_ds     = ScanpathDatasetWithPandas(data_dir=data_directory, 
									   name = 'val', 
									   transform=trans)
	val_loader = data.DataLoader(
		                     val_ds, batch_size = args.batch_size,
		                     shuffle = True,
		         			 collate_fn=pandas_collate_fn	
		                     )
		                     
	train_ds     = ScanpathDatasetWithPandas(data_dir=data_directory, 
									   name = 'train', 
									   transform=trans)
	train_loader = data.DataLoader(
		                     train_ds, batch_size = args.batch_size,
		                     shuffle = True,
		         			 collate_fn=pandas_collate_fn	
		                     )
		                     
	vocab_size= 32768
	encoder = EncoderCNN(args.embed_size)
	decoder = DecoderRNN(args.embed_size, args.hidden_size, 
					     vocab_size, args.num_layers)
	try:
		encoder.load_state_dict(torch.load(args.encoder_path))
		decoder.load_state_dict(torch.load(args.decoder_path))
		print("using pre-trained model")
	except:
		print("using new model")

	if torch.cuda.is_available():
		encoder.cuda()
		decoder.cuda()
	# Loss and Optimizer
	criterion = nn.CrossEntropyLoss()
	params = list(decoder.parameters()) + list(encoder.parameters())
	optimizer  = torch.optim.Adam(params, lr=args.learning_rate)
	total_step = len(train_loader)
	print('validating.....')
	best_val = validate(encoder, decoder, val_loader, criterion)
	print("starting val loss {:f}".format(best_val))
	
	for epoch in range(args.num_epochs):
		encoder.train()
		decoder.train()
		
		for batch_index, (images, targets, lengths) in enumerate(train_loader):
			# Set mini-batch dataset
			images = to_var(images, volatile=True)
			scanpaths = to_var(targets)
			scanpaths_packed = pack_padded_sequence(scanpaths, lengths, batch_first=True)[0]

			# Forward, Backward and Optimize
			decoder.zero_grad()
			encoder.zero_grad()
			#features = encoder(images)
			features = to_var(torch.zeros(args.batch_size,256))
			outputs  = decoder(features, scanpaths, lengths)

			loss = criterion(outputs, scanpaths_packed)
			loss.backward()
			optimizer.step()

			# Print log info
			if batch_index % args.log_step == 0:
				print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
					  %(epoch, args.num_epochs, batch_index, total_step, 
						loss.data[0], np.exp(loss.data[0]))) 

			# Save the models
		if (epoch+1) % args.save_step == 0:
			val_loss = validate(encoder, decoder, val_loader, criterion)
			print('val loss: ', val_loss)
			if val_loss < best_val:
				best_val = val_loss
				print("Found new best val")
				torch.save(decoder.state_dict(), 
						   os.path.join(args.model_path, 
										'decoder-%d-%d.pkl' %(15, 1)))
				torch.save(encoder.state_dict(), 
						   os.path.join(args.model_path, 
										'encoder-%d-%d.pkl' %(15, 1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' ,
                        help='path for saving trained models')
    parser.add_argument('--data_dir', type=str, default='data/' ,
                        help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=2,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1,
                        help='step size for saving trained models')
    parser.add_argument('--encoder_path', type=str, default='./models/encoder-15-1.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/decoder-15-1.pkl',
                        help='path for trained decoder')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--name', type=str, default='MIT1003')
    parser.add_argument('--split', type=float, default=0.9)
    
    args = parser.parse_args()
    main(args)
