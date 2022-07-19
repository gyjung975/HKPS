import numpy as np
import torch
import torch.nn as nn


class Tnet(nn.Module):
    def __init__(self, k=3):
        super(Tnet, self).__init__()
        self.k = k

        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1),
                                   nn.BatchNorm1d(1024),
                                   nn.ReLU())

        self.fc1 = nn.Sequential(nn.Linear(1024, 512),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 256),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU())
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, input):                   # input : (batch_size, features_dim; 3, #points; 1600)
        batch_size = input.size(0)

        x = self.conv1(input)                   # (batch_size, 64, 1600)
        x = self.conv2(x)                       # (batch_size, 128, 1600)
        x = self.conv3(x)                       # (batch_size, 1024 1600)

        pool = nn.MaxPool1d(x.size(-1))(x)      # (batch_size, 1024, 1)
        flat = nn.Flatten(1)(pool)              # (batch_size, 1024)

        x = self.fc1(flat)                      # (batch_size, 512)
        x = self.fc2(x)                         # (batch_size, 256)

        init = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1).cuda()
        # (batch_size, k, k)

        matrix = self.fc3(x).view(-1, self.k, self.k) + init    # (batch_size, k, k)
        return matrix


class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)

        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1),
                                   nn.BatchNorm1d(1024))

    def forward(self, input):                       # input : (batch_size, 3, 1600)
        matrix3x3 = self.input_transform(input)     # (batch_size, 3, 3)
        x = torch.bmm(input.transpose(1, 2), matrix3x3).transpose(1, 2)

        x = self.conv1(x)       # (batch_size, 64, 1600)

        matrix64x64 = self.feature_transform(x)     # (batch_size, 64, 64)
        x = torch.bmm(x.transpose(1, 2), matrix64x64).transpose(1, 2)

        x = self.conv2(x)       # (batch_size, 128, 1600)
        x = self.conv3(x)       # (batch_size, 1024, 1600)

        x = nn.MaxPool1d(x.size(-1))(x)             # (batch_size, 1024, 1)
        output = nn.Flatten(1)(x)                   # (batch_size, 1024)
        return output


class PointNet(nn.Module):
    def __init__(self, classes=15):
        super(PointNet, self).__init__()
        self.transform = Transform()

        self.fc1 = nn.Sequential(nn.Linear(1024, 512),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 256),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU())
        self.fc3 = nn.Linear(256, classes)

    def forward(self, input):           # input : (batch_size, 1024)
        x = self.transform(input)       # (batch_size, 512)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)            # (batch_size, classes)
        return output
