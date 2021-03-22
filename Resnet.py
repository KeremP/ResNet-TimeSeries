import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from GaussianNoise import GaussianNoise

class Resnet(nn.Module):
    def __init__(self, feat_cols, out_cols, drop_rates, hidden_units):
        super(Resnet, self).__init__()

        self.gauss = GaussianNoise() #inject Gausssian noise
        self.bn_0 = nn.BatchNorm1d(feat_cols)
        self.drop_0 = nn.Dropout(drop_rates[0])

        #layer 1
        self.lin_1 = nn.Linear(feat_cols,hidden_units)
        self.bn_1 = nn.BatchNorm1d(hidden_units)
        self.drop_1 = nn.Dropout(drop_rates[1])

        #layer 2
        self.lin_2 = nn.Linear(hidden_units+feat_cols,hidden_units)
        self.bn_2 = nn.BatchNorm1d(hidden_units)
        self.drop_2 = nn.Dropout(drop_rates[2])

        #layer 3
        self.lin_3 = nn.Linear(hidden_units+hidden_units, hidden_units)
        self.bn_3 = nn.BatchNorm1d(hidden_units)
        self.drop_3 = nn.Dropout(drop_rates[3])

        #layer 4
        self.lin_4 = nn.Linear(hidden_units+hidden_units, hidden_units)
        self.bn_4 = nn.BatchNorm1d(hidden_units)
        self.drop_4 = nn.Dropout(drop_rates[4])

        #output layer
        self.lin_5 = nn.Linear(hidden_units+hidden_units, out_cols)

        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.gauss(x)
        x = self.bn_0(x)
        x = self.drop_0(x)

        x1 = self.lin_1(x)
        x1 = self.bn_1(x1)
        x1 = self.Relu(x1)
        x1 = self.drop_1(x1)

        #concat residual blocks
        x = torch.cat([x,x1],1)

        x2 = self.lin_2(x)
        x2 = self.bn_2(x2)
        x2 = self.Relu(x2)
        x2 = self.drop_2(x2)

        x = torch.cat([x1,x2],1)

        x3 = self.lin_3(x)
        x3 = self.bn_3(x3)
        x3 = self.Relu(x3)
        x3 = self.drop_3(x3)

        x = torch.cat([x2,x3],1)

        x4 = self.lin_4(x)
        x4 = self.bn_4(x4)
        x4 = self.Relu(x4)
        x4 = self.drop_4(x4)

        x = torch.cat([x3,x4],1)

        x = self.lin_5(x)

        return x
