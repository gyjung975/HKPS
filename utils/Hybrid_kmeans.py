import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from numpy.linalg import norm
import copy


def cos_sim(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))


def distance(a, b):
    return sum([(el_a - el_b)**2 for el_a, el_b in list(zip(a, b))]) ** 0.5


def Kmeans_n(inputs, k, rate):      # inputs : (1600, 3)
    random_inputs = np.random.permutation(inputs)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(random_inputs)

    # estimate the normals of each point
    n = pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
    normals = np.asarray(pcd.normals)
    normals = random_inputs / abs(random_inputs) * abs(normals)     # (1600, 3)

    input_points = np.hstack([random_inputs, normals])              # (1600, 6)

    # kmeans++ for centroids
    X = KMeans(n_clusters=k, n_init=1, max_iter=1).fit(input_points)
    centroids = X.cluster_centers_                  # (k, 6)

    centroids_old = np.zeros(centroids.shape)       # (k, 6)
    labels = np.zeros(len(input_points))            # (1600, )

    error = np.zeros(k)                             # (k, )
    for i in range(k):
        error[i] = distance(centroids_old[i, :3], centroids[i, :3])

    for i in range(25):
        if np.mean(error) < 0.001:
            break

        for j in range(len(input_points)):      # 1600
            distances = np.zeros(k)             # (k, )

            for kk in range(k):
                distances[kk] = distance(input_points[j, :3], centroids[kk, :3]) \
                                + rate * cos_sim(input_points[j, 3:], centroids[kk, 3:])

            cluster = np.argmin(distances)
            labels[j] = cluster

        cos_losses = np.zeros(k)
        for j in range(k):
            points = [input_points[jj] for jj in range(len(input_points)) if labels[jj] == j]
            points = np.asarray(points)                 # (*, 6)
            centroids[j] = np.mean(points, axis=0)      # (6, )

            for kk in range(len(points)):
                cos_losses[j] += cos_sim(points[kk, 3:], centroids[j, 3:])

            cos_losses[j] = cos_losses[j] / len(points)
        cos_loss = np.max(cos_losses[~np.isnan(cos_losses)])

        centroids_old = copy.deepcopy(centroids)
        for j in range(k):
            error[j] = distance(centroids_old[j], centroids[j])

    return labels, input_points, cos_loss, centroids
    # (1600, )  /  (1600, 6)  /  ()  /  (k, 6)


def Kmeans(inputs, k, iteration=10, normals=True):
    labels_arr = np.zeros((iteration, len(inputs)))         # (iteration, 1600)
    outputs_arr = np.zeros((iteration, len(inputs), 6))     # (iteration, 1600, 6)
    cos_loss_arr = np.zeros(iteration)                      # (iteration)
    centroids_arr = np.zeros((iteration, k, 6))             # (iteration, k, 6)

    rate = 60 if normals else 0
    for i in range(iteration):
        labels_arr[i], outputs_arr[i], cos_loss_arr[i], centroids_arr[i] = Kmeans_n(inputs, k, rate=rate)

    idx = np.nanargmin(cos_loss_arr)

    labels = labels_arr[idx]
    outputs = outputs_arr[idx]
    cos_loss = cos_loss_arr[idx]
    centroids = centroids_arr[idx]

    return labels, outputs, cos_loss, centroids
    # (1600, )  /  (1600, 6)  /  ()  /  (k, 6)


if __name__ == "__main__":
    dataset = np.load('../dataset/with_noise.npy', allow_pickle=True)   # (67, 1600, 3)
    inputs = dataset[0]

    # Kmeans_n(inputs, k=11, rate=60)
    # Kmeans(inputs, k=11, iteration=3)
