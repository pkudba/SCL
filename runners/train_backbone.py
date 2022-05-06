import os
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
import time
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
from tools.utils import save_weights, get_total_parameters
from models.layer import EdgeConvolution, MultiEdgeConvolution, Pooler
from data.dataset import Train_ShapeNetPart
from models.siamese_net import Net
from models.model import Model, Tail, Encoder, Discriminator, ContrastiveLoss
from runners.base_run import Runner


class Trainer(Runner):
    def __init__(self,args):
        super(Trainer, self).__init__(args)

    def run(self, args):
        feeder = Train_ShapeNetPart(args, phase='train') 
        self.print_log("Train data is loading:...")
        train_loader = DataLoader(
                dataset=feeder,
                batch_size=args.train_batch_size*len(args.device),
                shuffle=True,
                num_workers=8 
            )
        
        testfd = Train_ShapeNetPart(args, phase='test') 
        self.print_log("Test data is loading:...")
        test_loader = DataLoader(
                dataset=testfd,
                batch_size=args.test_batch_size*len(args.device),
                shuffle=True,
                num_workers=8 
            )
        encoder = Encoder()
        encoder_para = get_total_parameters(encoder)
        self.print_log('Encoder:', encoder_para)

        md = Model()
        md = nn.DataParallel(md, device_ids=args.device)
        md = md.cuda(device=args.device[0])

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
            if epoch > 0 and epoch % 4 == 0:
                optimizer.param_groups[0]['lr'] *= 0.5
            self.print_log("learning rate : {}".format(optimizer.param_groups[0]['lr']))
            self.print_log("epoch {}:".format(epoch))
            
            for batch_id, (x, ro_coord, index, d_index, indexs, hard_neg) in enumerate(train_loader):
                index, d_index = index.cuda(device=args.device[0]), d_index.cuda(device=args.device[0])
                features = (md(x.cuda(device=args.device[0])))
                features = features.permute(0, 2, 1)
                ro_fea = (md(ro_coord.cuda(device=args.device[0])))
                ro_fea = ro_fea.permute(0, 2, 1)
                metric = ContrastiveLoss(device=args.device[0])
                loss = 0

                for t_dex, (pc, ro_pc, ind, d_ind, inds, hn) in enumerate(zip(features, ro_fea, index, d_index, indexs, hard_neg)):
                    out = md.module.gcn(pc, ind)
                    d_out = md.module.gcn(pc, d_ind)
                    ro_out = md.module.gcn(ro_pc, ind)
                    out = out[inds]
                    d_out = d_out[inds]
                    ro_out = ro_out[inds]
                    loss += metric(out, d_out, ro_out, hn, epoch)
                
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
    runner = Trainer()
    runner.run()


if __name__ == '__main__':
    main() 
    