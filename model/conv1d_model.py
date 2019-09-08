import torch
import torch.nn as nn
import torch.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.C1, self.D1 = make_sequential()
        self.C2, self.D2 = make_sequential()
        self.C3, self.D3 = make_sequential()
        self.C4, self.D4 = make_sequential()
        self.C5, self.D5 = make_sequential()
        self.C6, self.D6 = make_sequential()
        self.C7, self.D7 = make_sequential()

        self.CD1 = nn.Linear(7, 1)

    def forward(self, input):
        C1_input = input[:, 0, :].unsqueeze(1)
        C1_out = self.C1(C1_input)
        C1_out = C1_out.view(C1_out.shape[0], -1)
        C1_out = self.D1(C1_out)

        C2_input = input[:, 1, :].unsqueeze(1)
        C2_out = self.C2(C2_input)
        C2_out = C2_out.view(C2_out.shape[0], -1)
        C2_out = self.D2(C2_out)

        C3_input = input[:, 2, :].unsqueeze(1)
        C3_out = self.C3(C3_input)
        C3_out = C3_out.view(C3_out.shape[0], -1)
        C3_out = self.D3(C3_out)

        C4_input = input[:, 3, :].unsqueeze(1)
        C4_out = self.C4(C4_input)
        C4_out = C4_out.view(C4_out.shape[0], -1)
        C4_out = self.D4(C4_out)

        C5_input = input[:, 4, :].unsqueeze(1)
        C5_out = self.C5(C5_input)
        C5_out = C5_out.view(C5_out.shape[0], -1)
        C5_out = self.D5(C5_out)

        C6_input = input[:, 5, :].unsqueeze(1)
        C6_out = self.C6(C6_input)
        C6_out = C6_out.view(C6_out.shape[0], -1)
        C6_out = self.D6(C6_out)

        C7_input = input[:, 6, :].unsqueeze(1)
        C7_out = self.C7(C7_input)
        C7_out = C7_out.view(C7_out.shape[0], -1)
        C7_out = self.D7(C7_out)
        print(C7_out.shape)



        out = torch.cat([C1_out, C2_out, C3_out, C4_out, C5_out, C6_out, C7_out], dim=1)
        out = self.CD1(out)
        out = out.squeeze()
        return out

def make_sequential():
    cnn = nn.Sequential(nn.Conv1d(1, 10, 8, 4), # B, 10, 44
                        nn.MaxPool1d(2, 2), # B, 10, 22
                        nn.BatchNorm1d(10),
                        nn.ReLU(),
                        nn.Conv1d(10, 20, 4, 2), # B, 20, 10
                        nn.MaxPool1d(20, 2), # B, 20, 5
                        nn.BatchNorm1d(20),
                        nn.ReLU())

    linear = nn.Sequential(nn.Linear(20 * 5, 16),
                            nn.BatchNorm1d(16),
                            nn.ReLU(),
                            nn.Linear(16, 1))
    return cnn, linear