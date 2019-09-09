import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.C1, self.D1 = make_sequential()
        self.CD1 = nn.Linear(7, 1)
    def forward(self, input):
        C1_out = self.C1(input)
        C1_out = C1_out.view(C1_out.shape[0], -1)
        C1_out = self.D1(C1_out)
        C1_out = F.softmax(C1_out, dim=1)
        return C1_out
def make_sequential():
    cnn = nn.Sequential(nn.Conv1d(7, 10, 8, 4), # B, 10, 44
                        nn.MaxPool1d(2, 2), # B, 10, 22
                        nn.BatchNorm1d(10),
                        nn.ReLU(),
                        nn.Conv1d(10, 20, 4, 2), # B, 20, 10
                        nn.MaxPool1d(2, 2), # B, 20, 5
                        nn.BatchNorm1d(20),
                        nn.ReLU())
    linear = nn.Sequential(nn.Linear(20 * 5, 50),
                            nn.BatchNorm1d(50),
                            nn.ReLU(),
                            nn.Linear(50, 10))
    #cnn = nn.Sequential(nn.Conv1d(7, 20, 8, 4), # B, 10, 44
                        #nn.MaxPool1d(2, 2), # B, 10, 22
                        #nn.BatchNorm1d(20),
                        #nn.ReLU(),
                        #nn.Conv1d(20, 50, 4, 2), # B, 20, 10
                        #nn.MaxPool1d(2, 2), # B, 20, 5
                        #nn.BatchNorm1d(50),
                        #nn.ReLU())
    #linear = nn.Sequential(nn.Linear(50 * 5, 100),
                            #nn.BatchNorm1d(100),
                            #nn.ReLU(),
                            #nn.Linear(100, 10))
    return cnn, linear
