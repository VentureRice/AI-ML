# 载入需要的库
import pandas as pd
import numpy as np
import torch as th
from sklearn.model_selection import train_test_split 
import random
import re
import torch.nn.functional as F


# 读取数据
files1 = pd.read_csv("/home/user/venture/homework3/laptop-train.txt",sep="\t",header=None)
files2 = pd.read_csv("/home/user/venture/homework3/laptop-test.txt",sep="\t",header=None)

files1.columns = ['id','text','aspect','label']
files2.columns = ['id','text','aspect','label']

files1 = files1.loc[files1['label']!='conflict']
files2 = files2.loc[files2['label']!='conflict']

files1.replace('positive',0,inplace=True)
files1.replace('neutral',1,inplace=True)
files1.replace('negative',2,inplace=True)

files2.replace('positive',0,inplace=True)
files2.replace('neutral',1,inplace=True)
files2.replace('negative',2,inplace=True)


train_data,valid_data = train_test_split(files1,test_size = 0.3,
                                         random_state = 1234)


from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('/home/user/venture/homework3/glove.6B.300d.txt')
tmp_file = get_tmpfile("test_word2vec.txt")
_ = glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)

def get_aspect(aspect_str):
    a = [x.start() for x in re.finditer('\:',aspect_str)][0]
    return ' '+aspect_str[:a]

def get_phrase(data):
    aspect_str = data['aspect'].values[0]
    aspect = data['aspect'].apply(get_aspect)
    phrase = data['text'].values+aspect
    phrase = phrase.values
    return phrase

train_phrase = get_phrase(train_data)
valid_phrase = get_phrase(valid_data)
test_phrase = get_phrase(files2)

num_words = 300000
def get_list(phrase,num_words):
    train_list = []
    for i in range(len(phrase)):
        x = re.sub('\?|\.|\.|\,|\'|\,|\’','',phrase[i])
        x = x.lower()
        sentence = x.split(' ')
        ans = []
        for word in sentence:
            try:
                if model.vocab[word].index<num_words:
                    ans.append(model.vocab[word].index)
                else:
                    ans.append(num_words-1)
            except KeyError:
                ans.append(0)  
        
        train_list.append(ans)
    return train_list
    
train_list = get_list(train_phrase,num_words)
valid_list = get_list(valid_phrase,num_words)
test_list = get_list(test_phrase,num_words)

len_test_list = [len(x) for x in test_list]
len_valid_list = [len(x) for x in valid_list]


len_train_list = [len(x) for x in train_list]
embedding_dim = 300
embedding_matrix = np.zeros((num_words,embedding_dim))
for i in range(num_words):
    embedding_matrix[i,:] = model[model.index2word[i]]
embedding_matrix = th.tensor(embedding_matrix.astype('float32'))

class TextCNN(th.nn.Module):   
    def __init__(self, embedding_matrix,num_channels):
        super(TextCNN, self).__init__()
        self.embedding = th.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)#vocab_size*embedding dim
        self.conv0 = th.nn.Conv2d(in_channels=1,out_channels=num_channels,kernel_size=(2,embedding_dim),stride=1,padding=0)
        self.conv1 = th.nn.Conv2d(in_channels=1,out_channels=num_channels,kernel_size=(3,embedding_dim),stride=1,padding=0)
        self.conv2 = th.nn.Conv2d(in_channels=1,out_channels=num_channels,kernel_size=(4,embedding_dim),stride=1,padding=0)
        self.conv3 = th.nn.Conv2d(in_channels=1,out_channels=num_channels,kernel_size=(5,embedding_dim),stride=1,padding=0)
        self.conv4 = th.nn.Conv2d(in_channels=1,out_channels=num_channels,kernel_size=(6,embedding_dim),stride=1,padding=0)
        self.conv5 = th.nn.Conv2d(in_channels=1,out_channels=num_channels,kernel_size=(7,embedding_dim),stride=1,padding=0)
        self.conv6 = th.nn.Conv2d(in_channels=1,out_channels=num_channels,kernel_size=(8,embedding_dim),stride=1,padding=0)
        #self.conv7 = th.nn.Conv2d(in_channels=1,out_channels=num_channels,kernel_size=(9,embedding_dim),stride=1,padding=0)
        
        self.dropout = th.nn.Dropout(0.3)
        self.fc = th.nn.Linear(7*num_channels,3)
    
    @staticmethod
    def conv_block(x, conv):
        x = conv(x)
        x = F.relu(x.squeeze(3))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self, input_batch):
        embedded = self.embedding(input_batch)
        x = embedded.unsqueeze(1)
        
        x0 = self.conv_block(x,self.conv0)
        x1 = self.conv_block(x,self.conv1)
        x2 = self.conv_block(x,self.conv2)
        x3 = self.conv_block(x,self.conv3)
        x4 = self.conv_block(x,self.conv4)
        x5 = self.conv_block(x,self.conv5)
        x6 = self.conv_block(x,self.conv6)
        #x7 = self.conv_block(x,self.conv7)

        x = th.cat((x0,x1,x2,x3,x4,x5,x6),1)

        x = self.dropout(x)

        prob = F.softmax(self.fc(x),dim=1)
        return prob


def get_batch(train_list,label,batch_size):
    p = random.sample([x for x in range(len(train_list))], batch_size)
    p.sort(key=lambda x: len(train_list[x]), reverse=True)
    batch_index = [train_list[x] for x in p]
    batch_label = [label[x] for x in p]
    lens = [len(index) for index in batch_index]
    max_len = max(lens)
    batch_index = [index + [0] * (max_len - len(index)) for index in batch_index]
    return batch_index,batch_label,lens

num_channels = 30
encoder = TextCNN(embedding_matrix,num_channels)
optimizer = th.optim.Adam(encoder.parameters(), lr=0.001)
loss_func = th.nn.CrossEntropyLoss()

#device = th.device('cuda:0')
#encoder.to(device)

train_label = train_data['label'].values
valid_label = valid_data['label'].values


batch_valid_index,batch_valid_label,valid_lens = get_batch(valid_list,valid_label,len(valid_list))

batch_size = 512
loss_list = []
for epoch in range(300):   
    batch_index,batch_label,lens = get_batch(train_list,train_label,batch_size)
    optimizer.zero_grad()
    #output = encoder(th.tensor(batch_index,dtype = th.long).cuda())
    #loss = loss_func(output,th.tensor(batch_label,dtype = th.long).cuda())
    output = encoder(th.tensor(batch_index,dtype = th.long))
    loss = loss_func(output,th.tensor(batch_label,dtype = th.long))
    
    loss_list.append(loss) 
    if epoch%20 == 0:
        #output_valid = encoder(th.tensor(batch_valid_index,dtype = th.long).cuda())
        output_valid = encoder(th.tensor(batch_valid_index,dtype = th.long))
        pred = th.argmax(output_valid,dim=1)
        #acc = th.eq(pred, th.tensor(batch_valid_label,dtype = th.long).cuda()).sum().float().item()/len(batch_valid_label)
        acc = th.eq(pred, th.tensor(batch_valid_label,dtype = th.long)).sum().float().item()/len(batch_valid_label) 
        print("epoch = %.0f,loss = %.4f,acc on valid data = %.4f"%(epoch,loss,acc))
    loss.backward()
    optimizer.step()

test_index = [x for x in range(len(test_list))]
test_index.sort(key=lambda x:len(test_list[x]),reverse = True)
batch_test_index = [test_list[x] for x in test_index]
test_lens = [len(x) for x in batch_test_index]
max_len = max(test_lens)
batch_test_index = [index + [0] * (max_len - len(index)) for index in batch_test_index]

test_label = files2['label'].values
test_label = [test_label[x] for x in test_index]

#output_test = encoder(th.tensor(batch_test_index,dtype = th.long).cuda())
output_test = encoder(th.tensor(batch_test_index,dtype = th.long))
test_pred = th.argmax(output_test,dim=1)

#acc = th.eq(test_pred, th.tensor(test_label,dtype = th.long).cuda()).sum().float().item()/len(test_label) 
acc = th.eq(test_pred, th.tensor(test_label,dtype = th.long)).sum().float().item()/len(test_label)  
print(acc)

#test_pred = test_pred.cpu()

index_dic = {}

for i in range(len(test_pred)):
    index_dic[test_index[i]]=np.array(test_pred)[i]
    
ans = []
for i in range(len(index_dic)):
    ans.append(index_dic[i])

files1.replace(0,'positive',inplace=True)
files1.replace(1,'neutral',inplace=True)
files1.replace(2,'negative',inplace=True)
files2.replace(0,'positive',inplace=True)
files2.replace(1,'neutral',inplace=True)
files2.replace(2,'negative',inplace=True)

#test_label = files2['label']
#test_label.to_csv('gold.csv',index=0)

ans = pd.DataFrame(ans)

ans.replace(0,'positive',inplace=True)
ans.replace(1,'neutral',inplace=True)
ans.replace(2,'negative',inplace=True)

ans.to_csv('/home/user/venture/homework3/result_cnn.csv',index=0,header=None)
