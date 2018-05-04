# adapted from https://github.com/yunjey/show-attend-and-tell/blob/master/evaluate_model.ipynb
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from attention_data_loader import *
from extract_feature import get_transform
from attention_models import Decoder, EncoderVGG
from helper import *
import pickle
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (16.0, 12.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

model_path   = 'models/vgg_model.pth' 
encoder_path = 'models/attn_model.pth' # the trained model

dataloader =  get_loader(data_dir='data', 
                          name = 'train', 
                          feature_path = 'data/mit1003.pth', 
                          batch_size=2,
                          shuffle=False, 
                          num_workers=2)

encoder = EncoderVGG(model_path)
decoder = torch.load(encoder_path)

encoder.eval()
decoder.eval()


names = []
captions = []
alphas = []
for i, (image, name) in enumerate(dataloader):
    image = to_var(image)
    fea = encoder(image)
    fea = fea.view(fea.size(0), 512, 196).transpose(1, 2)
    ids, weights = decoder.sample(fea)
    names.append(name)
    captions.append(ids)
    alphas.append(weights)
    break
    if (i+1)%20 == 0:
        break
print(alphas)
"""
idx = 16
alps = torch.cat(alphas[idx][1:], 0)
cap = decode_captions(captions[idx].data.cpu().view(1, -1), vocab.idx2word)[0]
print(cap)
attention_visualization(root, names[idx][0], cap, alps.data.cpu())
idx += 1
"""
