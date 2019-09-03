import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from datasets import train_datasets
from model.dense_model import Model
from datasets import test_datasets
from tensorboardX import SummaryWriter

def train(learning_rate, nepoch, nepoch_summary, nepoch_model, save_path, load_path):
    train_loader = train_datasets()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Model().to(device)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    sepoch = 1

    if load_path != "":
        model, learning_rate, optimizer, sepoch = load_model(load_path, model, optimizer)
        print("Load model: ", load_path)
        sepoch += 1

    for epoch in range(sepoch, nepoch + 1):
        for i, data in enumerate(train_loader):
            inputs, lables = data
            inputs, lables = Variable(inputs).float().to(device), Variable(lables).float().to(device)

            y_pred = model(inputs)

            loss = criterion(y_pred, lables)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % nepoch_summary == 0:  # 매 10 iteration마다
            accuracy(model)
            write_summary(epoch, loss)
        if epoch % nepoch_model == 0:
            save_model(model, optimizer, learning_rate, epoch, save_path)



def accuracy(model):

    criterion = nn.MSELoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inputs, lables = test_datasets()
    inputs, lables = np.array(inputs), np.array(lables)
    inputs, lables = torch.from_numpy(inputs), torch.from_numpy(lables)
    inputs, lables = Variable(inputs).float().to(device), Variable(lables).float().to(device)

    y_pred = model(inputs)
    y_pred = y_pred.round()

    loss = torch.sqrt(criterion(y_pred, lables))
    print('Test Accuracy:', loss.item())

def write_summary(epoch, loss):
    summary.add_scalar('loss/loss', loss.item(), epoch)
    print("Write Summary")

def load_model(load_path, model, optimizer):
    load_dict = torch.load(load_path)
    model.load_state_dict(load_dict['model'])
    optimizer.load_state_dict(load_dict['optimizer'])
    learning_rate = load_dict['learning_rate']
    epoch = load_dict['epoch']

    print('Load model: ', load_path)

    return model, optimizer, learning_rate, epoch


def save_model(model, optimizer, learning_rate, epoch, save_path):
    save_path = save_path + 'iteration_' + str(epoch) + '.pth'
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer,
                'learning_rate': learning_rate,
                'epoch': epoch}, save_path)
    print("Save Model")

if __name__ == "__main__":
    learning_rate = 0.0001

    nepoch = 10000
    nepoch_summary = 50
    nepoch_model = 500

    save_path = "./output/"
    load_path = ""
    summary = SummaryWriter()

    train(learning_rate, nepoch, nepoch_summary, nepoch_model, save_path, load_path)
