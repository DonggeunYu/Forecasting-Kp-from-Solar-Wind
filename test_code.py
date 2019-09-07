import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

input_size = 10 # input dimension (word embedding) D
hidden_size = 30 # hidden dimension H
batch_size = 3
length = 7

rnn = nn.GRU(input_size,hidden_size,num_layers=1,bias=True,batch_first=True,bidirectional=True)
input = Variable(torch.randn(batch_size,length,input_size)) # B,T,D (3, 4, 10)
hidden = Variable(torch.zeros(2,batch_size,hidden_size)) # 2,B,H (2, 3, 30)

output, hidden = rnn(input, hidden)

print(output.size())
print(hidden.size())

import torch
from torch import nn

a1 = Variable(torch.randn(32, 1))
a1 = a1.squeeze()
print(a1.shape)