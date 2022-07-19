import numpy as np
from utils import Hybrid_kmeans


def dist_dataset(dataset_1, dataset_2):
    for i in range(len(dataset_1)):
        for j in range(len(dataset_2)):
            if Hybrid_kmeans.distance(dataset_1[i, :3], dataset_2[j, :3]) < 4:
                return True
    return False


def PlaneMerge(dataset, labels, k):
    # dataset : (1600, 6)
    # labels : (1600, )
    # k : optimal #planes of data

    relabels = np.array(range(k))       # (k, ) : (0, 1, ..., k - 1)

    for re in range(k + 2):
        k = int(max(labels) + 1)        # the number of planes of data

        for i in range(k):
            for j in range(k):
                mask_array = labels == i                    # (1600, ) : i-th cluster index
                dataset_1 = dataset[mask_array]             # (*, 6) : i-th cluster points
                centroids_1 = np.mean(dataset_1, axis=0)    # (6, ) : i-th cluster centroid

                mask_array = labels == j                    # (1600, ) : j-th cluster index
                dataset_2 = dataset[mask_array]             # (**, 6) : j-th cluster points
                centroids_2 = np.mean(dataset_2, axis=0)    # (6, ) : j-th cluster centroid

                if (i != j) and (Hybrid_kmeans.cos_sim(centroids_1[3:], centroids_2[3:]) < 0.1):
                    merge = dist_dataset(dataset_1, dataset_2)      # True or False

                    if merge:
                        relabels[j] = relabels[i]

        for i in range(k):
            relabels[i] = relabels[relabels[i]]

        for i in range(len(labels)):
            labels[i] = relabels[int(labels[i])]

    print("Finish Merge")

    count = 0
    for i in range(20):
        num = 0

        for j in range(len(labels)):
            num = j
            if labels[j] == i:
                break

        if num != (len(labels) - 1):
            for j in range(len(labels)):
                if labels[j] == i:
                    labels[j] = count
            count += 1

    return labels
