import torch as th
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import numpy as np
import matplotlib.pyplot as plt

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(16, 250),
                                     nn.ReLU(True),
                                     nn.Linear(250, 500),
                                     nn.ReLU(True),
                                     nn.Linear(500, 1000),
                                     nn.ReLU(True),
                                     nn.Linear(1000, 28*28),
                                     nn.Tanh())
 
    def forward(self, x, batch_size):

        x = self.decoder(x)
        x = x.reshape(batch_size,1,28,28)
        return x


decoder = Decoder()

#decoder = TheModelClass(*args, **kwargs)
decoder.load_state_dict(th.load('/home/user/venture/homework4/decoder.pth'))

vec = np.load('/home/user/venture/homework4/encoded.npy')
vec = th.tensor(vec)
vec = vec[:20,:]
output = decoder(vec,20).reshape(20,1,28,28)

plt.figure(figsize=(10, 10))
for i in range(20):
    x = output[i]
    plt.subplot(5, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x.squeeze(0).detach(), cmap=plt.cm.binary)

plt.savefig('/home/user/venture/homework4/imgs_generated.jpg',dpi=500,bbox_inches = 'tight')