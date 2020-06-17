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

train_dataset = MNIST(root='/home/user/venture/homework4/',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = MNIST(root='/home/user/venture/homework4/',train=False,transform=transforms.ToTensor())

if not os.path.exists('/home/user/venture/homework4/img'):
    os.mkdir('/home/user/venture/homework4/img')
 
 
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 100
batch_size = 512
learning_rate = 5e-4
 
#img_transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#])
 
#dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
 
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #self.encoder = nn.Sequential(nn.Linear(28*28, 256),
        #                             nn.ReLU(True),
        #                             nn.Linear(256, 64),
        #                             nn.ReLU(True),
        #                             nn.Linear(64, 12),
        #                             nn.ReLU(True),
        #                             nn.Linear(12, 2))
        self.encoder = nn.Sequential(nn.Linear(28*28, 1000),
                                     nn.ReLU(True),
                                     nn.Linear(1000, 500),
                                     nn.ReLU(True),
                                     nn.Linear(500, 250),
                                     nn.ReLU(True),
                                     nn.Linear(250, 16))
    def forward(self, x,batch_size):
        x = x.reshape(batch_size,1,-1)
        x = self.encoder(x)
        return x
        
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
 
    def forward(self, x,batch_size):

        x = self.decoder(x)
        x = x.reshape(batch_size,1,28,28)
        return x

#device = th.device("cuda" if th.cuda.is_available() else "cpu")
encoder = Encoder()
decoder = Decoder()


criterion = nn.MSELoss()
optimizer1 = th.optim.Adam(encoder.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
optimizer2 = th.optim.Adam(decoder.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
 
for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = Variable(img)
        # ===================forward=====================
        encoded = encoder(img,batch_size)
        output = decoder(encoded,batch_size)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss))
    if epoch % 10 == 0:
        pic = to_img(output.data)
        save_image(pic, '/home/user/venture/homework4/img/image_{}.png'.format(epoch))

th.save(encoder.state_dict(), '/home/user/venture/homework4/encoder.pth')
th.save(decoder.state_dict(), '/home/user/venture/homework4/decoder.pth')