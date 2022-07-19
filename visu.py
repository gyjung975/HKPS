import numpy as np
import argparse
import open3d as o3d

# dgcnn = o3d.io.read_point_cloud('../dgcnn.pytorch/visualization/area_1/conferenceRoom_1/conferenceRoom_1_gt.ply')
# print(dgcnn, '\n')


def visualization(args):
    num = args.data_num
    file = "./result/result_HKPS_" + str(num) + ".txt"

    coo = np.loadtxt(fname=file, delimiter=';', dtype='float')[:, :3]
    col = np.loadtxt(fname=file, delimiter=';', dtype='float')[:, 3:]

    min_ = col.min(0)
    max_ = col.max(0)
    for i in range(3):
        col[:, i] = (col[:, i] - min_[i]) / (max_[i] - min_[i])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coo)
    pcd.colors = o3d.utility.Vector3dVector(col)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--data_num', type=int, default=1)
    args = parser.parse_args()
    print(args)

    visualization(args)
