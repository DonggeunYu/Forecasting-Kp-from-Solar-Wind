import torch
import numpy as np
from torch.utils.data import Dataset
from datasets.data_loader import train_data

class Dataset(Dataset):
    def __init__(self):
        solar_wind, kp = train_data()
        self.train_data = np.array(solar_wind)
        self.train_lable = np.array(kp)

        self.length = len(self.train_lable)
        self.train_data = torch.from_numpy(self.train_data)
        self.train_lable = torch.from_numpy(self.train_lable)

    def __getitem__(self, item):
        return self.train_data[item], self.train_lable[item]

    def __len__(self):
        return self.length