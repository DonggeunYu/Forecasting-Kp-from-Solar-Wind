#date,kp_0h,kp_3h,kp_6h,kp_9h,kp_12h,kp_15h,kp_18h,kp_21h

import matplotlib.pyplot as plt

def kp():
    f = open('data/kp.csv')

    time = []
    kp_0h, kp_3h, kp_6h, kp_9h, kp_12h, kp_15h, kp_18h, kp_21h = [], [], [], [], [], [], [], []
    stop = 0

    for i, lines in enumerate(f.readlines()):
        line = lines.split(',')
        time_split = line[0].split('-')
        print(int(time_split[0] + time_split[1] + time_split[2]))
        # time.append(int(time_split[0] + time_split[1] + time_split[2]))
        time.append(i)
        kp_0h.append(line[1])
        kp_3h.append(line[2])
        kp_6h.append(line[3])
        kp_9h.append(line[4])
        kp_12h.append(line[5])
        kp_15h.append(line[6])
        kp_18h.append(line[7])
        kp_21h.append(line[8])

        stop += 1
        # if stop == 10000:
        # break

    plt.plot(time, kp_0h)

    plt.show()

def solar_wind():
    f = open('data/solar-wind/ace_1999.csv')
    stop = 0

    time = []
    Np, Tp, Vp, B_gsm_x, B_gsm_y, B_gsm_z, Bmag = [], [], [], [], [], [], []

    zero = 0

    for i, lines in enumerate(f.readlines()):
        line = lines.split(' ')
        #print(line)

        temp_arr = []
        for j, item in enumerate(line):
            if item == '':
                zero = 1
            else:
                temp_arr.append(float(item))
                #print(temp_arr)

        print(temp_arr)
        # doy,h,m,Np,Tp,Vp,B_gsm_x,B_gsm_y,B_gsm_z,Bmag
        n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = temp_arr
        Np.append(n4)
        Tp.append(n5)
        Vp.append(n6)
        B_gsm_x.append(n7)
        B_gsm_y.append(n8)
        B_gsm_z.append(n9)
        Bmag.append(n10)
        time.append(i)
        stop += 1
        if stop == 2:
            break
    plt.plot(time, B_gsm_z)
    plt.show()

solar_wind()