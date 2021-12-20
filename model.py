import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3,128,5,padding=2)
        self.pool = nn.MaxPool2d(2,stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(128,64,3,padding=1)
        self.flatten = Flatten()
        self.linear1 = nn.Linear(64*8*8,1024)
        self.linear2 = nn.Linear(1024,512)
        self.W = torch.randn(512,10) / math.sqrt(512*10)
        self.b = torch.zeros(10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.linear2(self.relu(self.linear1(x))))
        x = torch.matmul(x, self.W) + self.b
        return x

    def add_node(self, num):
        w = torch.randn(512, num) / math.sqrt(512*num)
        b = torch.zeros(num)
        self.W = torch.cat((self.W, w), dim=1)
        self.b = torch.cat((self.b, b), dim=0)
