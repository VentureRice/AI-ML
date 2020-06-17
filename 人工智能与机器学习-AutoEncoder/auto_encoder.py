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
from sklearn.manifold import TSNE

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 1000),
                                     nn.ReLU(True),
                                     nn.Linear(1000, 500),
                                     nn.ReLU(True),
                                     nn.Linear(500, 250),
                                     nn.ReLU(True),
                                     nn.Linear(250, 16))
    def forward(self, x, batch_size):
        x = x.reshape(batch_size,1,-1)
        x = self.encoder(x)
        return x

encoder = Encoder()

#decoder = TheModelClass(*args, **kwargs)
encoder.load_state_dict(th.load('/home/user/venture/homework4/encoder.pth'))

test_dataset = MNIST(root='/home/user/venture/homework4/',train=False,transform=transforms.ToTensor())
dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
test = Variable(test_dataset.data[9000:])
batch_size = 1000

for i,data in enumerate(dataloader):
    if i==9:
        img, label = data
        img = Variable(img)
        encoded = encoder(img,batch_size)


encoded = encoded.squeeze(1).detach()
tsne = TSNE(n_components=2,random_state=0)
reduce_feat = tsne.fit_transform(encoded)
x = reduce_feat[:,0]
y = reduce_feat[:,1]

plt.scatter(x,y,c=label.detach(),label=label.detach())
plt.savefig('/home/user/venture/homework4/encoder.jpg',dpi=500,bbox_inches = 'tight')

np.save('/home/user/venture/homework4/encoded',encoded.detach().numpy())