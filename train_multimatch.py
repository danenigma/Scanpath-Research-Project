import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pickle
from models import EncoderCNN, DecoderRNN 
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from data_loader import *
from multimatch_data_loader import *
import scipy.io as sio

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()    
def main(args):

	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

	batch_size  = args.batch_size
	train_data  = sio.loadmat('./data/scanpaths.mat')['target_scanpaths'].T
	labels      = sio.loadmat('./data/multimatch_target.mat')['output']
	table       = sio.loadmat('./data/multimatch_input.mat')['input']


	print(train_data.shape, labels.shape)
	train_ds     = MultiMatchDataset(train_data,table, labels)
	train_loader = data.DataLoader(
		     train_ds, batch_size = batch_size,
		     sampler = RandomSampler(train_ds)
		     )

	model = MultiMatchLoss()
	print(model)
	criterion = nn.MSELoss()
	try:
		model.load_state_dict(torch.load(args.model_path))
		print("using pre-trained model")
	except:
		print("using new model")

	if torch.cuda.is_available():
		model.cuda()
		criterion.cude()

	params     = model.parameters()
	#print(list(params))
	optimizer  = torch.optim.Adam(params, lr=args.learning_rate)
	total_step = len(train_loader)
	#https://discuss.pytorch.org/t/how-to-implement-accumulated-gradient-in-pytorch-i-e-iter-size-in-caffe-prototxt/2522/4
	points = []
	for epoch in range(5):
		train_loader_iter = iter(train_loader)	
		for i in range(args.num_epochs):
			optimizer.zero_grad() 
			batch_loss_value = 0
			if i*args.update > 20000-args.update:
				break
			for m in range(args.update):
				(seq_1, seq_2, target) = train_loader_iter.next()
				seq_1  = to_var(seq_1)
				seq_2  = to_var(seq_2)
				target = to_var(target)
				out    = model(seq_1, seq_2)
				loss   = criterion(out, target)
				loss.backward()
				batch_loss_value += loss.cpu().data.numpy()[0]

			optimizer.step()
			batch_loss_value = batch_loss_value/args.update
			if i % args.log_step == 0:
				print('Epoch [%d/%d],  Loss: %.4f'
				%(i, args.num_epochs, batch_loss_value)) 
				points.append(batch_loss_value)
	showPlot(np.array(points))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/mulimodel-1-1.pkl' ,
                        help='path for saving trained models')
    parser.add_argument('--data_dir', type=str, default='data/' ,
                        help='directory for resized images')
    parser.add_argument('--log_step', type=int , default=50,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1,
                        help='step size for saving trained models')
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--name', type=str, default='MIT1003')
    parser.add_argument('--split', type=float, default=0.9)
    parser.add_argument('--update', type=int, default=32)
    
    args = parser.parse_args()
    main(args)
