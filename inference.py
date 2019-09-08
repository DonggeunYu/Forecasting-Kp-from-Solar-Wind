import torch
import numpy as np
from torch.autograd import Variable
from openpyxl import Workbook
from model.conv1dv2_model import Model

def time_error(x, y):
    x = int(x)
    y = int(y)

    if x > y:
        return x - y - 1
    else:
        return 59 - y + x



def load_data(file_path):
    Np, Tp, Vp, B_gsm_x, B_gsm_y, B_gsm_z, Bmag, output = [], [], [], [], [], [], [], []
    turm = -1

    f = open(file_path)
    for line in f.readlines():
        line_splite = list(map(float, line.split(",")[2:]))

        Np.append(line_splite[1])
        Tp.append(line_splite[2])
        Vp.append(line_splite[3])
        B_gsm_x.append(line_splite[4])
        B_gsm_y.append(line_splite[5])
        B_gsm_z.append(line_splite[6])
        Bmag.append(line_splite[7])

        for i in range(time_error(line_splite[0], turm)):
            Np.append(Np[-1])
            Tp.append(Tp[-1])
            Vp.append(Vp[-1])
            B_gsm_x.append(B_gsm_x[-1])
            B_gsm_y.append(B_gsm_y[-1])
            B_gsm_z.append(B_gsm_z[-1])
            Bmag.append(Bmag[-1])

        turm = line_splite[0]

    for i in range(int(len(Np) / 3 / 60)):
        output.append([Np[i * 3 * 60:(i+1) * 3 * 60], Tp[i * 3 * 60:(i+1) * 3 * 60],
                       Vp[i * 3 * 60:(i+1) * 3 * 60], B_gsm_x[i * 3 * 60:(i+1) * 3 * 60],
                       B_gsm_y[i * 3 * 60:(i+1) * 3 * 60], B_gsm_z[i * 3 * 60:(i+1) * 3 * 60],
                       Bmag[i * 3 * 60:(i+1) * 3 * 60]])

    return output

def load_model(file_path):
    model = Model()

    load_dict = torch.load(file_path, map_location='cpu')
    model.load_state_dict(load_dict['model'])

    return model

def input_model(model, output):
    output = np.array(output)
    output = torch.from_numpy(output)
    output = Variable(output).float()

    y_pred = model(output)
    y_pred = np.argmax(y_pred.detach(), axis=1)

    y_pred = y_pred.numpy()

    return y_pred

def save_excel(y_pred):
    write_wb = Workbook()
    write_ws = write_wb.active

    i = 0
    j = 0

    for index in y_pred:
        write_ws.cell(i + 1, j + 1, index)
        j += 1
        if j == 8:
            i += 1
            j = 0

    write_wb.save('output.xlsx')

if __name__ == "__main__":

    output = load_data("data/problem.csv")
    print('Load data')

    model = load_model("output/iteration_2000.pth")
    print('Load model')

    y_pred = input_model(model, output)
    print('Input model')

    save_excel(y_pred)
    print('Save excel')