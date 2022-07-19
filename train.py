from utils import Hybrid_kmeans, Pointnet, Normalize, Make_labels

import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


def train(epochs, make_labels=False, save=True):
    device = torch.device('cuda')

    model = Pointnet.PointNet()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # an unorganized point cloud in the form of Nx3 (xyz) preprocessed by voxel downsammpling
    dataset = np.load('./dataset/with_noise.npy', allow_pickle=True)    # (67, 1600, 3)

    if make_labels:
        Make_labels.make_labels(dataset, max_k=15, iteration=10)

    dataset_normal = Normalize.normalize(dataset=dataset)                   # (67, 1600, 3)
    labels_ = np.load('./dataset/HKPS_labels.npy', allow_pickle=True)       # (67, )

    inputs_ = torch.tensor(dataset_normal).to(device).float()
    labels_ = torch.tensor(labels_).to(device).long()

    print("labels :", labels_ + 1)
    print("################")
    print("Train PointNet\n")

    for epoch in range(epochs):
        shuffle_idx = np.random.permutation(len(labels_))

        inputs_ = inputs_[shuffle_idx]
        labels_ = labels_[shuffle_idx]

        model.train()
        correct = 0

        iteration = int(len(inputs_) / 20)              # batch_size = 20

        for i in range(iteration):                      # 0 ~ 2
            inputs = inputs_[20 * i:20 * (i + 1)]       # (20, 1600, 3)
            labels = labels_[20 * i:20 * (i + 1)]       # (20)

            outputs = model(inputs.transpose(1, 2))     # (20, classes; optimal k for each size)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = outputs.argmax(1)
            correct += (pred == labels).sum()
        acc = 100 * correct / len(dataset)
        print('[train] [Epoch: %d] [loss: %.3f] [acc: %.2f]' % (epoch + 1, loss.item(), acc))

        model.eval()
        correct_val = 0

        inputs_val = inputs_[:20]
        labels_val = labels_[:20]

        outputs_val = model(inputs_val.transpose(1, 2))
        loss_val = criterion(outputs_val, labels_val)

        pred_val = outputs_val.argmax(1)
        correct_val += (pred_val == labels_val).sum()
        acc_val = 100 * correct_val / len(dataset)

        print("[valid] [Epoch: %d] [loss: %.3f] [acc: %.2f]\n" % (epoch + 1, loss_val.item(), acc_val))

        if save:
            torch.save(model.state_dict(), './model_save/save_' + str(epoch) + '.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HKPS')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--make_labels', type=bool, default=False,
                        help='Make labels of data (optimal k of each data)')
    parser.add_argument('--save', type=bool, default=True,
                        help='Saving Model')
    args = parser.parse_args()
    print(args)

    if not os.path.exists('model_save'):
        os.mkdir('model_save')

    train(epochs=args.epochs, make_labels=args.make_labels, save=args.save)
