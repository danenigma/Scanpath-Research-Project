import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from feat_data_loader import get_loader
import argparse
from torch.autograd import Variable

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
    

def get_model(model_path=None):
	if model_path is None:
		vgg   = models.vgg16(pretrained=True)
		model = nn.Sequential(*(vgg.features[i] for i in range(29)))
		torch.save(model, 'models/vgg_model.pth')
	else:
		model = torch.load(model_path)

	if torch.cuda.is_available():
		model = model.cuda()	

	return model


def get_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    return transform


def extract_features(data_dir, transform, batch_size, shuffle, num_workers, model):

	dataloader = get_loader(data_dir, transform, batch_size, shuffle, num_workers)
	model.eval()

	features = []
	n_iters = len(dataloader)
	for i, images in enumerate(dataloader):
		images = to_var(images)
		feas   = model(images).cpu()
		features.append(feas.data)

		if (i+1)%100 == 0:
		    print('iter [%d/%d] finsihed.'%(i, n_iters))
	return torch.cat(features, 0)#, imnames


def main(args):

	data_dir  = args.data_dir
	transform = get_transform()
	batch_size = args.batch_size
	shuffle = args.shuffle
	num_workers = args.num_workers
	model = get_model()#args.model_path)

	features = extract_features(data_dir, transform, 
								batch_size, shuffle, 
								num_workers, model)
	print(features.shape)							
	torch.save(features, args.save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='file lists')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle the dataset')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of threads for data loader')
    parser.add_argument('--model_path', type=str, default='models/vgg_model.pth',
                        help='The path to the feature extraction model')
    parser.add_argument('--save_name', type=str, default='data/mit1003.pth', help='Where to save the files.')
    args = parser.parse_args()
    print(args)
    main(args)
    print('Done.')




