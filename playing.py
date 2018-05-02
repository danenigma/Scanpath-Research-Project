import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

class VGG16(nn.Module):

    def __init__(self):
        """Load the pretrained VGG-16 and replace top fc layer."""
        super(VGG16, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.model = nn.Sequential(*(self.vgg.features[i] for i in range(29)))
       
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.self.model(images)

        return features
if __name__=='__main__':
	vgg = VGG16()
	x   = Variable(torch.rand(1, 3, 224, 224))
	print('out: ', vgg.model(x).shape)
