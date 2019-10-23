#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from snownlp import SnowNLP

output_path='/home/guohf/AI_tutorial/ch8/output_files/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Dataloader():
    def __init__(self):
        #path= tf.keras.utils.get_file('nietzsche.txt',
        #                             origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        path = '/home/guohf/old_man_and_sea.txt'
        with open(path, encoding='utf-8') as f:
            self.raw_text= f.read().lower()
            self.vocablary= SnowNLP(self.raw_text)
        self.chars= sorted(list(set(self.vocablary.words)))
        print("The number of chars:",len(self.chars))
        self.char_indices= dict((c,i) for i, c in enumerate(self.chars))
        self.indices_char= dict((i,c) for i, c in enumerate(self.chars))
        self.text= [self.char_indices[c] for c in self.vocablary.words]

    def get_batch(self, batch_size):
        first_word= []
        next_word= []
        for i in range(batch_size):
            index= np.random.randint(0, len(self.text)-3-batch_size)
            first_word.append(self.text[index:index+3])
            next_word.append(self.text[index+3])
        return np.array(first_word), np.array(next_word)

class NN(nn.Module):
    def __init__(self, num_chars, batch_size, seq_size):
        super(NN, self).__init__()
        self.num_chars= num_chars
        self.batch_size= batch_size
        self.seq_size= seq_size
        self.embedding= nn.Embedding(self.num_chars, 100)
        self.fc1 = nn.Sequential(
            nn.Linear(self.seq_size*self.num_chars*100, 250),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(250, self.num_chars)

    # 定义前向传播过程，输入为inputs
    def forward(self, inputs):
        inputs= torch.LongTensor(inputs)
        inputs= torch.reshape(inputs,(self.batch_size*self.seq_size, 1))
        inputs= torch.zeros(self.batch_size*self.seq_size, self.num_chars).scatter_(1, inputs, 1) # one-hot encoding
        inputs= torch.reshape(inputs, (self.batch_size, self.seq_size, self.num_chars))
        x= self.embedding(inputs.long())
        x= x.view(x.size(0), -1)
        x = self.fc1(x)
        outputs = self.fc2(x)
        return outputs

    def predict(self, inputs, temperature=1.):
        inputs = torch.LongTensor(inputs)
        inputs = torch.reshape(inputs, (self.seq_size, 1))
        inputs = torch.zeros(self.seq_size, self.num_chars).scatter_(1, inputs, 1)  # one-hot encoding
        inputs = torch.reshape(inputs, (1, self.seq_size, self.num_chars))
        x = self.embedding(inputs.long())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        predicts = self.fc2(x)
        prob = F.softmax(predicts/ temperature).detach().numpy()
        return np.array([np.random.choice(self.num_chars, p=prob[0, :])])

dataloader = Dataloader()
batch_size= 50
seq_size= 3
learning_rate= 0.001
EPOCH= 50000

net = NN(len(dataloader.chars), batch_size, seq_size).to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer= optim.Adam(net.parameters(), lr= learning_rate)


# 训练
if __name__ == "__main__":
    start_time= time.time()
    for epoch in range(EPOCH):
        sum_loss = 0.0
        # 数据读取
        inputs, labels = dataloader.get_batch(batch_size)

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, torch.LongTensor(labels))
        loss.backward()
        optimizer.step()
        end_time= time.time()
        print('the %d epoch loss: %.03f, total running time: %.2f s'% (epoch, loss.item(),end_time-start_time))


    torch.save(net.state_dict(),'/home/guohf/AI_tutorial/ch8/model/oldman_vocbased_%d.pt'%EPOCH)
    fout=open(output_path+str(EPOCH)+"_voc_based_output.txt", "w")

    X_, _=dataloader.get_batch(1)
    for diversity in [0.2,0.5,1.0,1.2]:
        X=X_
        print("diversity %f:" % diversity)
        fout.write("diversity %f:\n" % diversity)
        for t in range(100):
            y_pred = net.predict(X,diversity)
            fout.write(dataloader.indices_char[y_pred[0]])
            print(dataloader.indices_char[y_pred[0]], end='',flush=False)
            #print("shape of X:", X)
            X=np.concatenate([X[:,1:], np.expand_dims(y_pred, axis=1)], axis=-1)
        print("/n")
        fout.write("\n")
    fout.close()

