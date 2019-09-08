import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.L1 = nn.Linear(7 * 3 * 60, 7 * 4 * 60)
        self.B1 = nn.BatchNorm1d(7 * 4 * 60)
        self.L2 = nn.Linear(7 * 4 * 60, 4 * 60)
        self.B2 = nn.BatchNorm1d(4 * 60)
        self.L3 = nn.Linear(4 * 60, 60)
        self.B3 = nn.BatchNorm1d(60)
        self.L4 = nn.Linear(60, 10)
        self.B4 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = F.relu(self.B1(self.L1(x)))
        x = F.relu(self.B2(self.L2(x)))
        x = F.relu(self.B3(self.L3(x)))
        x = self.L4(x)

        x = F.softmax(x, dim=1)

        return x
