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
	val_size = len(data_loader)
	val_loss = 0
	encoder.eval()
	decoder.eval()

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
		val_loss += loss.data.sum()
		
	return val_loss/val_size
    
def main(args):

	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

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
			  
	#labels = np.concatenate(labels)
	print('labels shape: ', labels.shape)
	print('img data: ', img_data.shape)
	#split = int(img_data.shape[0]*args.split)
	#concat_split = int(concat_labels.shape[0]*args.split)

	#print('split: ', split, 'concat split: ', concat_split)

	#train_data = img_data[:split]
	#val_data   = img_data[split+1:]

	#train_labels = labels[:split]
	#val_labels   = labels[split+1:]
	 
	vocab  = np.load(
			 os.path.join(
			 args.data_dir,
			 '{}-vocab.npy'.format(args.name)))

	vocab_size  = vocab.reshape(1,-1).shape[1]			 

	train_size  = img_data.shape[0]
	#data_table  = np.concatenate(np.array([[x]*15 for x in range (train_size)]))
	#np.random.shuffle(data_table)
	#print('table: ', data_table, data_table.shape)
	#train_scanpath_ds = ScanpathDatasetWithTable(img_data, labels, data_table, vocab)
	train_scanpath_ds = ScanpathDataset(img_data[103:], labels[103:], vocab)
	val_scanpath_ds   = ScanpathDataset(img_data[:103], labels[:103], vocab)

	train_data_loader = data.DataLoader(
					             train_scanpath_ds, batch_size = args.batch_size,
					             sampler = RandomSampler(train_scanpath_ds),
					             collate_fn = collate_fn)
	val_data_loader = data.DataLoader(
					             val_scanpath_ds, batch_size = args.batch_size,
					             sampler = RandomSampler(val_scanpath_ds),
					             collate_fn = collate_fn)

	#val_scanpath_ds = ScanpathDataset(val_data, val_labels, vocab)

	#val_data_loader = data.DataLoader(
	#				             val_scanpath_ds, batch_size = args.batch_size,
	#				             sampler = RandomSampler(val_scanpath_ds),
	#				             collate_fn = collate_fn)
					             
	print(len(train_scanpath_ds), len(val_scanpath_ds))
	encoder = EncoderCNN(args.embed_size)
	decoder = DecoderRNN(args.embed_size, args.hidden_size, 
					     vocab_size, args.num_layers)
	#torch.save(decoder.state_dict(), 
	#		   os.path.join(args.model_path, 
	#						'decoder-%d-%d.pkl' %(15, 1)))
	#torch.save(encoder.state_dict(), 
	#		   os.path.join(args.model_path, 
	#						'encoder-%d-%d.pkl' %(15, 1)))
	#print('saving done')
	#return
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
	params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
	optimizer = torch.optim.Adam(params, lr=args.learning_rate)
	total_step = len(train_data_loader)
	print('validating.....')
	best_val = validate(encoder, decoder, val_data_loader, criterion)
	print("starting val loss {:f}".format(best_val))

	for epoch in range(args.num_epochs):
		encoder.train()
		decoder.train()
		for i, (images, targets, saliencies, lengths) in enumerate(train_data_loader):

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
			#break
			# Save the models
		if (epoch+1) % args.save_step == 0:
			val_loss = validate(encoder, decoder, val_data_loader, criterion)
			print('val loss: ', val_loss)
#			if val_loss < best_val:
#				best_val = val_loss
#			print("Found new best val")
			torch.save(decoder.state_dict(), 
					   os.path.join(args.model_path, 
									'decoder-%d-%d.pkl' %(15, 1)))
			torch.save(encoder.state_dict(), 
					   os.path.join(args.model_path, 
									'encoder-%d-%d.pkl' %(15, 1)))
		#break
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' ,
                        help='path for saving trained models')
    parser.add_argument('--data_dir', type=str, default='data/' ,
                        help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=2,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=5,
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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--name', type=str, default='MIT1003')
    parser.add_argument('--split', type=float, default=0.9)
    
    args = parser.parse_args()
    main(args)
