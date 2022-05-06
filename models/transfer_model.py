import sys
sys.path.append("..")
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import OrderedDict
from models.transfer_layer import EdgeConvolution, MultiEdgeConvolution, Pooler
import random
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math

class Encoder(nn.Module):
    def __init__(self, k=20):
        super(Encoder, self).__init__()
        self.conv0 = MultiEdgeConvolution(k, in_features=6, mlp=(64, 64))
        self.conv1 = MultiEdgeConvolution(k, in_features=64, mlp=(64, 64))
        self.conv2 = EdgeConvolution(k, in_features=64, out_features=64)
        self.conv3 = EdgeConvolution(k, in_features=64, out_features=64)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        features = torch.cat((x1, x2, x3, x4), dim=1)  
        return features


class Tail(nn.Module):
    def __init__(self, in_features=256):
        super(Tail, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(in_features, 128, kernel_size=1, bias=False)),
            ('bn0', nn.BatchNorm1d(128)),
            ('relu0', nn.LeakyReLU(negative_slope=0.2))
        ]))

    def forward(self, x):
        x = self.conv(x)
        return x


class Model(nn.Module):
    def __init__(self, k=20, out_features=3):
        super(Model, self).__init__()
        self.encoder = Encoder(k=k)
        self.tail = Tail()
        self.connected_layer = nn.Sequential(OrderedDict([
            ('Linear0', nn.Linear(in_features=256, out_features=286, bias=False)),
            ('bn0', nn.BatchNorm1d(286)),
            ('relu0', nn.LeakyReLU(negative_slope=0.2)),
            ('Linear2', nn.Linear(in_features=286, out_features=8, bias=False)),
        ]))
        self.gcn = GCNConv(128, 128)
    
    def forward(self, *args):
        if len(args) == 2:
            x, y = args[0], args[1]
            x1 = self.tail(self.encoder(x))
            x2 = self.tail(self.encoder(y))
            x = torch.cat((x1, x2), dim=1) 
            matrix = self.decoder(x)
            return matrix
        elif len(args) == 1:
            x = args[0]
            features = self.tail(self.encoder(x))
            return features
        else:
            raise ValueError('Invalid number of arguments.')


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class ContrastiveLoss(object):
    def __init__(self, device, negative_samples=1024, temp=0.07):
        self.device = device
        self.negative_samples = negative_samples
        self.temp = temp

    def calculate_sample(self, original, transformed, negative):
        positive_similarity = F.cosine_similarity(original, transformed) / self.temp
        negative_similarity = F.cosine_similarity(original, negative) / self.temp
        similarities = torch.cat((positive_similarity, negative_similarity))
        loss = F.cross_entropy(
            similarities[None, :], torch.tensor([0]).to(self.device)
        )
        return loss

    def __call__(self, original, dilated, transformed, hb, cur_epoch): 
        batch_size = original.shape[0]
        losses = 0.0
        if cur_epoch<8:
            w_l = 0.3
            w_u = 1
        elif cur_epoch<28:
            w_l = 0.3 + 0.02*(cur_epoch-7)
            w_u = 1 - 0.02*(cur_epoch-7)
        else:
            w_l = 0.5
            w_u = 0.75

        for i in range(batch_size):
            hard_nega = []
            for j in range(batch_size):
                if hb[i][j]<w_u and hb[i][j]>w_l:
                    hard_nega.append(j)
            if len(hard_nega)==0:
                hard_nega.append(0)
            nega_sample = transformed[hard_nega,:]     
            posi_sample = dilated[None, i, :]
            loss = self.calculate_sample(
                original[None, i, :], posi_sample, nega_sample
            )
            losses += loss
        return losses / batch_size
