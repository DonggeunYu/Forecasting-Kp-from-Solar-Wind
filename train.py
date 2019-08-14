import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from datasets import train_datasets
from model import dense_model
from datasets import test_datasets
from tensorboardX import SummaryWriter

def train(nepoch, nepoch_summary, nepoch_model, save_path):
    train_loader = train_datasets()

    model = dense_model.Model()

    criterion = nn.MSELoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    for epoch in range(nepoch):
        for i, data in enumerate(train_loader):
            inputs, lables = data
            inputs, lables = Variable(inputs).float(), Variable(lables).float()

            y_pred = model(inputs)

            loss = criterion(y_pred, lables)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy(model)
        if epoch % nepoch_summary == 0:  # 매 10 iteration마다
            summary_write(epoch, loss)
        if epoch % nepoch_model == 0:
            model_save(epoch, save_path, model)



def accuracy(model):

    criterion = nn.MSELoss()

    inputs, lables = test_datasets()
    inputs, lables = np.array(inputs), np.array(lables)
    inputs, lables = torch.from_numpy(inputs), torch.from_numpy(lables)
    inputs, lables = Variable(inputs).float(), Variable(lables).float()

    y_pred = model(inputs)
    y_pred = y_pred.round()

    loss = torch.sqrt(criterion(y_pred, lables))
    print('Test Accuracy:', loss)

def summary_write(epoch, loss):
    summary.add_scalar('loss/loss', loss.item(), epoch)
    print("Write Summary")

def model_save(epoch, save_path, model):
    save_path = save_path + 'epoch' + str(epoch) + '.pth'
    torch.save(model.state_dict(), save_path)
    print("Save Model")

if __name__ == "__main__":
    nepoch = 10
    nepoch_summary = 2
    nepoch_model = 2
    save_path = "./output/"
    summary = SummaryWriter()

    train(nepoch, nepoch_summary, nepoch_model, save_path)