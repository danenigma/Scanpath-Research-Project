import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

class ResNet50(nn.Module):

	def __init__(self, embed_size):
		"""Load the pretrained VGG-16 and replace top fc layer."""
		super(ResNet50, self).__init__()
		model = models.resnet50(pretrained=False)
		modified_model = list(model.children())
		modified_model[0] = nn.Conv2d(1, 64, 
									  kernel_size=(7, 7), 
									  stride=(2, 2), 
									  padding=(3, 3),
									  bias=False) 
		self.modified_model = nn.Sequential(*(modified_model[:-1]))
		self.linear = nn.Linear(2048, embed_size)
		self.bn     = nn.BatchNorm1d(embed_size, momentum=0.01)
		self.init_weights()
	
	def init_weights(self):
		"""Initialize the weights."""
		self.linear.weight.data.normal_(0.0, 0.02)
		self.linear.bias.data.fill_(0)
		
	def forward(self, images):
		"""Extract the image feature vectors."""
		features = self.modified_model(images)
		features = Variable(features.data)
		features = features.view(features.size(0), -1)
		features = self.bn(self.linear(features))
		return features
		
if __name__=='__main__':
	resnet = ResNet50(512)
	x      = Variable(torch.rand(2, 1, 224, 224))
	print('out: ', resnet(x).shape)
