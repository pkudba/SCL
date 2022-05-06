import sys
sys.path.append("..")
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import OrderedDict
from models.siamese_layer import EdgeConvolution, MultiEdgeConvolution, Pooler
import tools.utils
import random
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Encoder(nn.Module):
    def __init__(self, k=20):
        super(Encoder, self).__init__()
        self.conv0 = MultiEdgeConvolution(k, in_features=6, mlp=(64, 64))
        self.conv1 = MultiEdgeConvolution(k, in_features=64, mlp=(64, 128))

    def forward(self, x):
        x1 = self.conv0(x)
        return x1


class Tail(nn.Module):
    def __init__(self, in_features=64):
        super(Tail, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(in_features, 32, kernel_size=1, bias=False)),
            ('bn0', nn.BatchNorm1d(32)),
            ('relu0', nn.LeakyReLU(negative_slope=0.2))
        ]))

    def forward(self, x):
        x = self.conv(x)
        return x


class Net(nn.Module):
    def __init__(self, k=20, out_features=3):
        super(Net, self).__init__()
        self.encoder = Encoder(k=k)
        self.tail = Tail()
        self.connected_layer = nn.Linear(in_features=64, out_features=2, bias=False)
        self.gcn = GCNConv(32, 32)
     
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
            similarities[None, :], torch.tensor([0]).to('cuda')
        )
        return loss

    def __call__(self, original, transformed, negative): 
        batch_size = original.shape[0] 
        losses = 0.0
        for i in range(batch_size):
            nega_sample = torch.cat((negative[0:i, :], negative[i:batch_size, :]))
            posi_sample = torch.cat((transformed[None, i, :], negative[None, i, :]))
            loss = self.calculate_sample(
                original[None, i, :], posi_sample, nega_sample
            )
            losses += loss
        return losses / batch_size
