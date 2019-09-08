import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from datasets import train_datasets
from model.conv1dv2_model import Model
from datasets import test_datasets
from tensorboardX import SummaryWriter
from time import localtime, strftime

def train(learning_rate, nepoch, nepoch_summary_a, nepoch_summary, nepoch_model, save_path, load_path,
          batch_size, shuffle, numworkers):
    train_loader = train_datasets(batch_size=batch_size, shuffle=shuffle, numworkers=numworkers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Model().to(device)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    sepoch = 1

    if load_path != "":
        model, learning_rate, optimizer, sepoch = load_model(load_path, model, optimizer)
        model.to(device)
        print("Load model: ", load_path)
        sepoch += 1

    for epoch in range(sepoch, nepoch + 1):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).float().to(device), Variable(labels).long()

            num_classes = 10
            labels = torch.eye(num_classes)[labels].to(device)

            y_pred = model(inputs)

            loss = criterion(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % nepoch_summary_a == 0:
            accuracy(epoch, model)
        if epoch % nepoch_summary == 0:  # 매 10 iteration마다
            write_summary(epoch, loss)
        if epoch % nepoch_model == 0:
            save_model(model, optimizer, learning_rate, epoch, save_path)


def accuracy(epoch, model):
    criterion = nn.MSELoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inputs, labels = test_datasets()
    inputs, labels = np.array(inputs), np.array(labels)
    inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
    inputs, labels = Variable(inputs).float().to(device), Variable(labels).long()

    num_classes = 10
    labels = torch.eye(num_classes)[labels].to(device)

    y_pred = model(inputs)

    loss = torch.sqrt(criterion(y_pred, labels))
    print('Test Accuracy:', epoch, loss.item())

    write_summary_a(epoch, loss.item())


def write_summary_a(epoch, loss):
    summary.add_scalar('Accuracy/Accuracy', loss, epoch)
    #print("Write Summary")


def write_summary(epoch, loss):
    summary.add_scalar('Loss/Loss', loss.item(), epoch)
    #print("Write Summary")


def load_model(load_path, model, optimizer):
    load_dict = torch.load(load_path)
    model.load_state_dict(load_dict['model'])
    optimizer.load_state_dict(load_dict['optimizer'].state_dict())
    learning_rate = load_dict['learning_rate']
    epoch = load_dict['epoch']

    print('Load model: ', load_path)

    return model, learning_rate, optimizer, epoch


def save_model(model, optimizer, learning_rate, epoch, save_path):
    save_path = save_path + 'iteration_conv1dv2_1' + str(epoch) + '.pth'
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer,
                'learning_rate': learning_rate,
                'epoch': epoch}, save_path)
    print("Save Model")


if __name__ == "__main__":
    learning_rate = 0.00001

    nepoch = 100000
    nepoch_summary_a = 500
    nepoch_summary = 100
    nepoch_model = 1000

    batch_size = 2048
    shuffle = True
    numworkers = 0

    save_path = "./output/"
    load_path = ""
    summary = SummaryWriter('runs/' + 'conv1dv2_' + strftime("%Y-%m-%d-%H-%M/", localtime()))

    train(learning_rate, nepoch, nepoch_summary_a, nepoch_summary, nepoch_model, save_path, load_path,
          batch_size, shuffle, numworkers)
