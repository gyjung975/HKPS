import numpy as np


def normalize(dataset):
    xmin = []
    xmax = []

    for i in range(3):
        xmin.append(np.min(dataset[:, :, i]))       # x, y, z 축별 min
        xmax.append(np.max(dataset[:, :, i]))       # x, y, z 축별 max
    print(xmin)
    print(xmax)

    dataset_normal = np.zeros(dataset.shape)        # (67, 1600, 3)
    for i in range(3):
        dataset_normal[:, :, i] = (dataset[:, :, i] - xmin[i]) / (xmax[i] - xmin[i])

    return dataset_normal
