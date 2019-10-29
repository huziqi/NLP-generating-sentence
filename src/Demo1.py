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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
char2index= dict()
class Dataloader():
    def __init__(self):
        #path= tf.keras.utils.get_file('nietzsche.txt',
        #                             origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        path = '/home/guohf/old_man_and_sea.txt'
        with open(path, encoding='utf-8') as f:
            self.raw_text= f.read().lower()
        self.chars= sorted(list(set(self.raw_text)))
        self.char_indices= dict((c,i) for i, c in enumerate(self.chars))
        self.indices_char= dict((i,c) for i, c in enumerate(self.chars))
        self.text= [self.char_indices[c] for c in self.raw_text]
        char2index=self.char_indices

    def get_batch(self, batch_size):
        first_word= []
        next_word= []
        for i in range(batch_size):
            index= np.random.randint(0, len(self.text)-3-batch_size)
            first_word.append(self.text[index:index+3])
            next_word.append(self.text[index+3])
        return np.array(first_word), np.array(next_word)

    def get_index(self, char):
        return self.char_indices[char]

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
net = NN(len(dataloader.chars), batch_size, seq_size).to(device)


print("Please enter three words:")
inputs=input()
X=[]
sentence=[]
for i in range(3):
    sentence.append(inputs[i])
    X.append(dataloader.get_index(inputs[i]))
X = np.array(X)
for t in range(10):
    y_pred = net.predict(X,0.8)
    #print(dataloader.indices_char[y_pred[0]], end='',flush=False)
    X=np.concatenate([X[1:], y_pred], axis=-1)
    sentence.append(dataloader.indices_char[y_pred[0]])
print("自动生成的语句为: ", end='')
for i in range(len(sentence)):
    print(sentence[i],end='')
# else:
#     if input()=='v':
#         net.load_state_dict(torch.load('/home/guohf/AI_tutorial/ch8/model/oldman_vocbased_50000.pt'))
#         print("Please enter three vocabulrary:")
#         inputs = input()
#         X = []
#         for i in range(3):
#             X.append(char2index[inputs[i]])
#         for t in range(10):
#             y_pred = net.predict(X, 0.5)
#             print(dataloader.indices_char[y_pred[0]], end='', flush=False)
#             X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
#     else:
#         print("enter the wrong key, please enter 'w' or 'v'!")