import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class EncoderCNN(nn.Module):

    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
    
class ResNet50(nn.Module):

	def __init__(self, embed_size):
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

    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        return sampled_ids.squeeze()

class MultiMatchLoss(nn.Module):

	def __init__(self):
		super(MultiMatchLoss, self).__init__()

		self.lstm1   = nn.LSTM(3, 128)
		self.lstm2   = nn.LSTM(3, 128)
		
		self.layers = nn.ModuleList([
					  nn.Linear(256, 512),
			          nn.ReLU(),
					  nn.Linear(512, 512),
			          nn.ReLU(),
			          #nn.BatchNorm1d(1024),
					  nn.Linear(512, 256),
			          nn.ReLU(),
			          #nn.BatchNorm1d(256),
					  nn.Linear(256, 5),
					  nn.Sigmoid()
					  ])	
					  
	def forward(self, seq1, seq2):
		seq1_h, _   = self.lstm1(seq1)         
		seq2_h, _   = self.lstm2(seq2) 
		#print(seq1_h[0][-1].shape, seq2_h[0][-1].shape)
		x = torch.cat((seq1_h[0][-1], seq2_h[0][-1]), 0)
		for layer in self.layers:
			x = layer(x)
		return x



