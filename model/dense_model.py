import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.L1 = nn.Linear(7 * 3 * 60, 3 * 60)
        self.B1 = nn.BatchNorm1d(3 * 60)
        self.L2 = nn.Linear(3 * 60, 60)
        self.B2 = nn.BatchNorm1d(60)
        self.L3 = nn.Linear(60, 30)
        self.B3 = nn.BatchNorm1d(30)
        self.L4 = nn.Linear(30, 8)
        self.B4 = nn.BatchNorm1d(8)
        self.L5 = nn.Linear(8, 1)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = F.relu(self.B1(self.L1(x)))
        x = F.relu(self.B2(self.L2(x)))
        x = F.relu(self.B3(self.L3(x)))
        x = F.relu(self.B4(self.L4(x)))
        x = F.relu(self.L5(x))

        return x