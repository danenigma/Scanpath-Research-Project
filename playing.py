import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

class VGG16(nn.Module):

    def __init__(self):
        """Load the pretrained VGG-16 and replace top fc layer."""
        super(VGG16, self).__init__()
        self.vggnet = models.vgg16(pretrained=True)
        self.vggnet.classifier = nn.Sequential(
                                 nn.Linear(self.vggnet.classifier[0].in_features, 1024),
                                 nn.Linear(1024, 17))
 
            
        #self.init_weights()

    def init_weights(self):
        """Initialize the weights."""

        label_dist = torch.FloatTensor([0.47966632, 0.06534585, 0.08689607, 0.04066736, 0.05943691, 0.06047967,
        0.03267292, 0.04935697, 0.01668405, 0.02294056, 0.0132082,  0.01494612,
        0.01529371, 0.01668405, 0.00729927, 0.00868961, 0.00973236])

        self.vggnet.classifier.weight.data.uniform_(-0.1, 0.1)
        self.vggnet.classifier.bias.data = label_dist

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.vggnet(images)

        return features
if __name__=='__main__':
	vgg = VGG16()
	x   = Variable(torch.rand(1, 3, 512, 512))
	print('out: ', vgg.vggnet.features(x).shape)
