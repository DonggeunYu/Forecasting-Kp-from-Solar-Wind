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



def load_data():
    Np, Tp, Vp, B_gsm_x, B_gsm_y, B_gsm_z, Bmag = [], [], [], [], [], [], []
    for file in range(1999, 2014):
        f = open(('data/solar-wind/ace_' + str(file) + '.csv'))
        stop = 0

        time = []

        turm = -1
        iterations = -1

        for i, lines in enumerate(f.readlines()):
            line = lines.split(' ')
            # print(line)
            iterations += 1
            temp_arr = []
            for j, item in enumerate(line):
                if item != '':
                    temp_arr.append(float(item))
                    # print(temp_arr)

            # doy,h,m,Np,Tp,Vp,B_gsm_x,B_gsm_y,B_gsm_z,Bmag
            n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = temp_arr

            Np.append(n4)
            Tp.append(n5)
            Vp.append(n6)
            B_gsm_x.append(n7)
            B_gsm_y.append(n8)
            B_gsm_z.append(n9)
            Bmag.append(n10)
            time.append(iterations)

            '''
            x = int(n3)
            y = int(turm)

            if x > y:
                if x - y > 5:
                    print(n1, n2, n3)
            else:
                if 59 - y + x > 5:
                    print(n1, n2, n3)
            '''

            for i in range(time_error(n3, turm)):
                Np.append(Np[-1])
                Tp.append(Tp[-1])
                Vp.append(Vp[-1])
                B_gsm_x.append(B_gsm_x[-1])
                B_gsm_y.append(B_gsm_y[-1])
                B_gsm_z.append(B_gsm_z[-1])
                Bmag.append(Bmag[-1])

                iterations += 1
                time.append(iterations)

            turm = n3

            # if i == 100:
            # break

    output = []
    for i in range(int(len(Np) / 3 / 60)):
        output.append([Np[i * 3 * 60:(i + 1) * 3 * 60], Tp[i * 3 * 60:(i + 1) * 3 * 60],
                       Vp[i * 3 * 60:(i + 1) * 3 * 60], B_gsm_x[i * 3 * 60:(i + 1) * 3 * 60],
                       B_gsm_y[i * 3 * 60:(i + 1) * 3 * 60], B_gsm_z[i * 3 * 60:(i + 1) * 3 * 60],
                       Bmag[i * 3 * 60:(i + 1) * 3 * 60]])

    return output

def kp():
    f = open('data/kp.csv')

    time = []
    kp = []

    for i, lines in enumerate(f.readlines()):
        line = list(map(float, lines.split(',')[1:]))

        #time_split = line[0].split('-')
        #print(int(time_split[0] + time_split[1] + time_split[2]))
        # time.append(int(time_split[0] + time_split[1] + time_split[2]))
        time.append(i)
        for i in range(0, 8):
            kp.append(line[i])

        # if stop == 10000:
        # break
    #kp = kp[:2555]
    return kp

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

def RMSE(y_pred, labels):
    return np.sqrt(((y_pred - labels) ** 2).mean())

if __name__ == "__main__":
    model = load_model("output/Conv1dv2_1/iteration_9000.pth")
    print('Load model')

    output = load_data()
    print('Load data')
    print(np.shape(output))


    y_pred = input_model(model, output)
    print('Input model')

    labels = kp()
    print('Load Kp')

    accuracy = RMSE(y_pred, labels)
    print('RMSE: ', accuracy)

    # conv1dv2_1
    # 10000: 0.758
    # 9000: 0.753
    # 5000: 0.814