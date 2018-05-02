import numpy as np
import argparse
from torch.nn.utils.rnn import  pack_padded_sequence
from attention_models import *
from attention_data_loader import *
import torch.optim as optim

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def train(dataloader, model, optimizer, criterion, epoch, total_epoch):
	total_step = len(dataloader)
	# print 'Total step:', total_step
	for i, (features, targets, lengths) in enumerate(dataloader):
		optimizer.zero_grad()
		features = to_var(features).transpose(1,2)
		targets  = to_var(targets)
		predicts = model(features, targets[:, :-1], [l - 1 for l in lengths])
		predicts = pack_padded_sequence(predicts, [l-1 for l in lengths], batch_first=True)[0]
		targets  = pack_padded_sequence(targets[:, 1:], [l-1 for l in lengths], batch_first=True)[0]
		loss = criterion(predicts, targets)
		loss.backward()
		optimizer.step()
		#if (i+1)%10 == 0:
		print('Epoch [%d/%d]: [%d/%d], loss: %5.4f, perplexity: %5.4f.'%(epoch, total_epoch,i,
				                                                             total_step,loss.data[0],
				                                                             np.exp(loss.data[0])))

	
def test():
    pass


def main(args):
    # dataset setting

	feature_path = args.feature_path
	batch_size   = args.batch_size
	shuffle      = args.shuffle
	num_workers  = args.num_workers
	dataloader  =  get_loader(data_dir='data', 
			  				  name = 'train', 
			   				  feature_path = feature_path, 
			   				  batch_size=batch_size,
		       				  shuffle=shuffle, 
		       				  num_workers=num_workers)
		       				  
	vocab_size= 32768

	# model setting
	vis_dim = args.vis_dim
	vis_num = args.vis_num
	embed_dim = args.embed_dim
	hidden_dim = args.hidden_dim
	num_layers = args.num_layers
	dropout_ratio = args.dropout_ratio

	model = Decoder(vis_dim=vis_dim,
		            vis_num=vis_num, 
		            embed_dim=embed_dim,
		            hidden_dim=hidden_dim, 
		            vocab_size=vocab_size, 
		            num_layers=num_layers,
		            dropout_ratio=dropout_ratio)

	# optimizer setting
	lr = args.lr
	num_epochs = args.num_epochs
	optimizer = optim.Adam(model.parameters(), lr=lr)

	# criterion
	criterion = nn.CrossEntropyLoss()
	if torch.cuda.is_available():
		model = model.cuda()
		criterion = criterion.cuda()
	
	model.train()

	print('Number of epochs:', num_epochs)
	print(model)
	
	for epoch in range(num_epochs):
		train(dataloader=dataloader, model=model, optimizer=optimizer, criterion=criterion,
		      epoch=epoch, total_epoch=num_epochs)
		torch.save(model, 'models/attn_model.pth')
		


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data loader
    parser.add_argument('--feature_path', type=str,
                        default='data/mit1003.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=2)
    # model setting
    parser.add_argument('--vis_dim', type=int, default=512)
    parser.add_argument('--vis_num', type=int, default=196)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=155)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout_ratio', type=float, default=0.5)

    # optimizer setting
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=120)

    args = parser.parse_args()
    print(args)
    main(args)
