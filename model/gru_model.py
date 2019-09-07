import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, bias, batch_first, bidirectional):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        if bidirectional:
            self.num_bidirectional = 2
        else:
            self.num_bidirectional = 1

        self.GRU = nn.GRU(self.input_size, self.hidden_size, self.num_layers, self.bias, self.batch_first, bidirectional=self.bidirectional)
        self.L1 = nn.Linear(60, 30)
        self.B1 = nn.BatchNorm1d(30)
        self.L2 = nn.Linear(30, 1)

    def forward(self, x, hidden):
        out, hidden = self.GRU(x, hidden)
        out = out[:, -1, :]
        out = F.relu(self.B1(self.L1(out)))
        out = self.L2(out)
        out = out.squeeze()
        return out

    def init_hidden(self):
        return Variable(torch.zeros(self.num_layers * self.num_bidirectional, self.batch_size, self.hidden_size))