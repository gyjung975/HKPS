from utils import Hybrid_kmeans, Pointnet, Normalize, Merge, Visualize

import numpy as np
import argparse
import os
import torch
import warnings
warnings.filterwarnings('ignore')


def valid(data_len=10, save_num=99, iteration=10):
    device = torch.device('cuda')

    model = Pointnet.PointNet()
    model.to(device)

    dataset = np.load('./dataset/with_noise.npy', allow_pickle=True)
    dataset_normal = Normalize.normalize(dataset)[:data_len]
    inputs = torch.tensor(dataset_normal).to(device).float()        # (10, 1600, 3)

    file = "./model_save/save_" + str(save_num) + '.pth'
    model.load_state_dict(torch.load(file))

    model.eval()
    with torch.no_grad():
        outputs = model(inputs.transpose(1, 2))     # (10, classes)

    pred = outputs.argmax(1)        # (10)
    print('PointNet result (optimal k for each data) :', pred + 1)

    k = np.asarray(pred.cpu() + 1)
    for i in range(len(k)):
        print("%d / %d" % (i + 1, len(k)))

        labels_k, outputs_k, _, _ = Hybrid_kmeans.Kmeans(dataset[i], k[i], iteration, normals=True)

        labels_k = Merge.PlaneMerge(outputs_k, np.copy(labels_k), k[i])

        result_file = "HKPS_" + str(i)
        Visualize.visualize(labels_k, outputs_k, result_file)

    print("Result saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--data_len', type=int, default=10,
                        help='the number of data for test')
    parser.add_argument('--model_num', type=int,
                        help='Model number for test')
    parser.add_argument('--iteration', type=int, default=5)
    args = parser.parse_args()
    print(args)

    if not os.path.exists('result'):
        os.mkdir('result')

    valid(data_len=args.data_len, save_num=args.model_num - 1, iteration=args.iteration)
