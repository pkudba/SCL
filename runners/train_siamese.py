import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import OrderedDict
import random
import pytorch3d.ops
from torch_geometric.nn import GCNConv
from torch_geometric.transforms.knn_graph import knn_graph
from torch_geometric.nn.conv import MessagePassing 
from torch_sparse import matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import fps
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from sklearn import preprocessing

import sys
sys.path.append("..")
from models.siamese_layer import EdgeConvolution, MultiEdgeConvolution
from data.dataset import Siamese_ShapeNetPart
from tools.utils import import_class, get_total_parameters, save_weights
from runners.base_run import Runner
from models.siamese_net import Net, Tail, Encoder, Discriminator, ContrastiveLoss

        
class SiameseNetRunner(Runner):
    def __init__(self,args):
        super(SiameseNetRunner, self).__init__(args)
        self.loss = nn.NLLLoss().to(self.output_dev)

    def run(self, args):
        feeder = Siamese_ShapeNetPart(args, phase='train') 
        self.print_log("Train data is loading:...")
        train_loader = DataLoader(
                dataset=feeder,
                batch_size=args.train_batch_size*len(args.device),
                shuffle=True,
                num_workers=8 
            )

        encoder = Encoder()
        encoder_para = get_total_parameters(encoder)
        self.print_log('Encoder:', encoder_para)

        md = Net()
        md = nn.DataParallel(md, device_ids=args.device)
        md = md.cuda(device=args.device[0])
        func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(md.parameters(), lr=args.lr, weight_decay=0)
 
        losses = []
        for epoch in range(args.num_epochs):
            los = []
            los2 = []
            md.train()
            train_loss = 0
            lbl_1 = torch.ones(args.train_batch_size, args.nb_nodes)
            lbl_2 = torch.zeros(args.train_batch_size, args.nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)
            if epoch > 0 and epoch % 5 == 0:
                optimizer.param_groups[0]['lr'] *= 0.5
            self.print_log("learning rate : {}".format(optimizer.param_groups[0]['lr']))
            self.print_log("epoch {}:".format(epoch))
            
            for batch_id, (x, ro_coord, index, d_index, indexs) in enumerate(train_loader):
                index, d_index = index.cuda(device=args.device[0]), d_index.cuda(device=args.device[0])
                features = (md(x.cuda(device=args.device[0])))
                features = features.permute(0, 2, 1)
                ro_fea = (md(ro_coord.cuda(device=args.device[0])))
                ro_fea = ro_fea.permute(0, 2, 1)
                loss = 0                
                for t_dex, (pc, ro_pc, ind, d_ind, inds) in enumerate(zip(features, ro_fea, index, d_index, indexs)):
                    out = md.module.gcn(pc, ind)
                    d_out = md.module.gcn(pc, d_ind)
                    ro_out = md.module.gcn(ro_pc, ind)
                    out = out[inds]
                    d_out = d_out[inds]
                    ro_out = ro_out[inds]
                    idx = np.random.permutation(range(0,16))
                    neg_out = out[idx,:]
                    output_pos = md.module.connected_layer(torch.cat((out, ro_out), dim=-1))
                    output_neg = md.module.connected_layer(torch.cat((out, neg_out), dim=-1))
                    pos = torch.ones(16, dtype=int).to(args.device[0])
                    neg = torch.zeros(16, dtype=int).to(args.device[0])
                    loss += func(output_pos, pos)
                    loss += func(output_neg, neg)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                los.append(loss.item()) 
                train_loss = np.mean(los)
                if batch_id % 5 == 0:
                    self.print_log("the batch of num {}".format(batch_id))
                    self.print_log("train_loss: {}".format(train_loss))

            losses.append(train_loss) 
            
            save_weights(epoch, md, optimizer, r'./log/logs/backbone/{}.pt'.format(epoch))


def main():
    runner = SiameseNetRunner()
    runner.run()


if __name__ == '__main__':
    main() 