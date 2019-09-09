a = [1, 4, 3]

import numpy as np


def normalization(arr):
    min = np.min(arr)
    max = np.max(arr)
    temp = []
    for i in arr:
        temp.append((i - min) / (max - min))

    return temp

print(normalization(a))