from torch.utils.data import DataLoader
from datasets.dataset_loader import Dataset
from datasets.data_loader import test_data

def train_datasets(batch_size=32, shuffle=True, numworkers=0):
    datasets = Dataset()

    load_data = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=shuffle, num_workers=numworkers)

    return load_data

def test_datasets():
    return test_data()