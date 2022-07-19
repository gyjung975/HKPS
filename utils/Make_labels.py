import numpy as np
from utils import Hybrid_kmeans


def make_labels(data, max_k=15, iteration=10):
    # max_k : the maximum number that PointNet will estimate

    labels = np.zeros(len(data))        # (67, )
    for size in range(len(data)):
        print("make labels %d / %d :" % (int(size) + 1, len(data)))

        loss = np.zeros(max_k)
        for k in range(1, max_k + 1):   # 1 ~ 15
            # data[size] : (1600, 3)
            _, _, cos_loss, _ = Hybrid_kmeans.Kmeans(data[size], k, iteration, normals=True)

            loss[k - 1] = cos_loss

        labels[size] = max_k - 1
        for i in range(max_k - 1, 0, -1):
            if (loss[i - 1] - loss[i]) > 0.01:
                labels[size] = i
                break

        print(labels)
        print("label :", labels[size] + 1)
        np.save("../dataset/HKPS_labels.npy", labels)
