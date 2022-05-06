import os
import json
import torch
import random
import numbers
import math
import numpy as np
import torch.nn as nn

import sys
sys.path.append("..")
from tools.utils import load_h5
from tools.utils import load_h5_data_label_seg
from tools.utils import create_output

from torch_geometric.nn import fps
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.transforms.knn_graph import knn_graph
from torch_geometric.nn import GCNConv

import time
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from models.siamese_net import Net
from models.model import Model, Tail, Encoder, Discriminator, ContrastiveLoss
from runners.base_run import Runner


class Aggregator(MessagePassing):
    def __init__(self, k=20, **kwargs):
        super(Aggregator, self).__init__(aggr='add', **kwargs)
        self.k = k
        self.d_k = 2*k

    def forward(self, x):
        edge_index = knn_graph(x, self.k, None)
        dilated_index = knn_graph(x, self.d_k, None)[0, ::2]
        edge_index, edge_weight = gcn_norm(
            edge_index, None, x.size(0),
            improved=False, add_self_loops=True, dtype=x.dtype)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)


class KNN(MessagePassing):
    def __init__(self, k=160, **kwargs):
        super(KNN, self).__init__(aggr='add', **kwargs)
        self.k = k
        self.d_k = 2*k

    def forward(self, x):
        edge_index = knn_graph(x, self.k, None)
        dilated_index = knn_graph(x, self.d_k, None)[:, ::2]
        return edge_index, dilated_index


class Rotate(MessagePassing):
    def __init__(self, degrees=90, **kwargs):
        super(Rotate, self).__init__(aggr='add', **kwargs)
        self.degrees = degrees

    def forward(self, x, y):
        degree = math.pi * random.uniform(-self.degrees, self.degrees) / 180.0
        num = random.randint(0, 2)
        sin, cos = math.sin(degree), math.cos(degree)
        Matrix = torch.zeros([3, 3, 3])
        Matrix[0] = torch.tensor([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
        Matrix[1] = torch.tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
        Matrix[2] = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        x = torch.mm(x, Matrix[num])
        y = torch.mm(y, Matrix[num])
        return torch.cat((x, y), 1)


class ShapeNetPart(Dataset):
    def __init__(self, args, num_points=2048, ratio=0.0, num_pairs=512,
                 need_norm=True, phase='train'):
        self.knn = KNN()
        self.rotate = Rotate()
        self.path = args.data_path
        self.data_path = os.path.join(args.data_path, 'ShapeNet_hdf5_data')
        self.skeleton_path = os.path.join(
            args.data_path, 'ShapeNet_skeletons', phase
        )
        self.num_points = num_points
        self.num_classes = 16
        self.num_parts = 50
        self.num_pairs = num_pairs
        self.need_norm = need_norm
        self.phase = phase
        self.classes_dict = {
            'airplane': [0, 1, 2, 3],   # 0
            'bag': [4, 5],
            'cap': [6, 7],
            'car': [8, 9, 10, 11],
            'chair': [12, 13, 14, 15],
            'earphone': [16, 17, 18],
            'guitar': [19, 20, 21],
            'knife': [22, 23],
            'lamp': [24, 25, 26, 27],
            'laptop': [28, 29],
            'motorbike': [30, 31, 32, 33, 34, 35],
            'mug': [36, 37],
            'pistol': [38, 39, 40],
            'rocket': [41, 42, 43],
            'skateboard': [44, 45, 46],
            'table': [47, 48, 49]
        }
        self.shape_names = {}
        for cat in self.classes_dict.keys():
            for label in self.classes_dict[cat]:
                self.shape_names[label] = cat

        if phase == 'train':
            files = os.path.join(self.data_path, 'train_hdf5_file_list.txt')
        else:
            files = os.path.join(self.data_path, 'test_hdf5_file_list.txt')
        file_list = [line.rstrip() for line in open(files)]
        num_files = len(file_list) 

        self.coordinates = list()
        self.labels = list()
        self.segmentation = list()
        for i in range(num_files):
            cur_file = os.path.join(self.data_path, file_list[i])
            cur_data, cur_label, cur_seg = load_h5_data_label_seg(cur_file) 
            cur_data = cur_data[:, 0:self.num_points, :]
            cur_seg = cur_seg[:, 0:self.num_points]
            self.coordinates.append(cur_data)
            self.labels.append(cur_label)
            self.segmentation.append(cur_seg)  
        self.coordinates = np.vstack(self.coordinates).astype(np.float32)
        self.labels = np.vstack(self.labels).squeeze().astype(np.int64)
        self.segmentation = np.vstack(self.segmentation).astype(np.int64)

        self.siamese_model = Net()
        self.runner = Runner(args)
        self.runner.load_model_weights(self.siamese_model, './log/pretrained_models/siamese_model.pt')
        self.gcn = GCNConv(32, 32)

        color_map_file = os.path.join(self.data_path, 'part_color_mapping.json')
        self.color_map = json.load(open(color_map_file, 'r'))

    def __getitem__(self, index):
        coord = self.coordinates[index]  # N*3
        coord = torch.tensor(coord, dtype=torch.float32)
        label = self.labels[index]
        seg = self.segmentation[index]  # N
        one_hot = np.zeros(shape=(16, 1), dtype=np.int64)
        one_hot[label, 0] = 1

        if self.phase == 'train':
            indices = list(range(self.num_points))
            np.random.shuffle(indices)
            coord = coord[indices]
            seg = seg[indices]
        if self.need_norm:
            if self.phase == 'train':
                norm = np.load(os.path.join(self.path, 'train_npynorm_of_ShapeNet/3D index {}.npy'.format(index)))  
            if self.phase == 'test':
                norm = np.load(os.path.join(self.path, 'test_npynorm_of_ShapeNet/3D index {}.npy'.format(index)))  
            
            norm = torch.tensor(norm, dtype=torch.float32)
            
            coord = (torch.cat((coord, norm),1)).T   
            return coord, norm, one_hot, seg
        else:
            batch = torch.zeros([2048], dtype=torch.int64)
            indexs = fps(coord, batch, ratio=0.03125)
            dss = coord[indexs]
            labelr = self.labels[index]
            index, d_index = self.knn(coord)
            return coord.T, dss.T, coord.T, dss.T, one_hot, seg, index, d_index, indexs

    def __len__(self):
        return self.coordinates.shape[0]

    def output_colored_point_cloud(self, points, seg):
        output_file = 'color_part.ply'
        one = np.ones((points.shape[0], 3))
        for i in range(self.num_points):
            one[i] = self.color_map[seg[i]]
        create_output(points, one*255, output_file)


class Cla_ShapeNetPart(Dataset):
    def __init__(self, args, num_points=2048, ratio=0.0, num_pairs=512,
                 need_norm=True, phase='train'):
        self.knn = KNN()
        self.rotate = Rotate()
        self.path = args.data_path
        self.data_path = os.path.join(args.data_path, 'ShapeNet_hdf5_data')
        self.skeleton_path = os.path.join(
            args.data_path, 'ShapeNet_skeletons', phase
        )
        self.skeleton_path = os.path.join(
            args.data_path, 'ShapeNet_skeletons', phase
        )
        self.num_points = num_points
        self.num_classes = 16
        self.num_parts = 50
        self.num_pairs = num_pairs
        self.need_norm = need_norm
        self.phase = phase
        self.classes_dict = {
            'airplane': [0, 1, 2, 3],   # 0
            'bag': [4, 5],
            'cap': [6, 7],
            'car': [8, 9, 10, 11],
            'chair': [12, 13, 14, 15],
            'earphone': [16, 17, 18],
            'guitar': [19, 20, 21],
            'knife': [22, 23],
            'lamp': [24, 25, 26, 27],
            'laptop': [28, 29],
            'motorbike': [30, 31, 32, 33, 34, 35],
            'mug': [36, 37],
            'pistol': [38, 39, 40],
            'rocket': [41, 42, 43],
            'skateboard': [44, 45, 46],
            'table': [47, 48, 49]
        }
        self.shape_names = {}
        for cat in self.classes_dict.keys():
            for label in self.classes_dict[cat]:
                self.shape_names[label] = cat

        if phase == 'train':
            files = os.path.join(self.data_path, 'train_hdf5_file_list.txt')
        else:
            files = os.path.join(self.data_path, 'test_hdf5_file_list.txt')
        file_list = [line.rstrip() for line in open(files)]
        num_files = len(file_list) 

        self.coordinates = list()
        self.labels = list()
        self.segmentation = list()
        for i in range(num_files):
            cur_file = os.path.join(self.data_path, file_list[i])
            cur_data, cur_label, cur_seg = load_h5_data_label_seg(cur_file) 
            cur_data = cur_data[:, 0:self.num_points, :]
            cur_seg = cur_seg[:, 0:self.num_points]
            self.coordinates.append(cur_data)
            self.labels.append(cur_label)
            self.segmentation.append(cur_seg)  
        self.coordinates = np.vstack(self.coordinates).astype(np.float32)
        self.labels = np.vstack(self.labels).squeeze().astype(np.int64)
        self.segmentation = np.vstack(self.segmentation).astype(np.int64)

        self.siamese_model = Net()
        self.runner = Runner(args)
        self.runner.load_model_weights(self.siamese_model, './log/pretrained_models/siamese_model.pt')
        self.gcn = GCNConv(32, 32)

        color_map_file = os.path.join(self.data_path, 'part_color_mapping.json')
        self.color_map = json.load(open(color_map_file, 'r'))

    def __getitem__(self, index):
        coord = self.coordinates[index]  # N*3
        coord = torch.tensor(coord, dtype=torch.float32)
        label = self.labels[index]
        seg = self.segmentation[index]  # N
        one_hot = np.zeros(shape=(16, 1), dtype=np.int64)
        one_hot[label, 0] = 1

        if self.phase == 'train':
            indices = list(range(self.num_points))
            np.random.shuffle(indices)
            coord = coord[indices]
            seg = seg[indices]
        if self.need_norm:
            if self.phase == 'train':
                norm = np.load(os.path.join(self.path, 'train_npynorm_of_ShapeNet/3D index {}.npy'.format(index)))  
            if self.phase == 'test':
                norm = np.load(os.path.join(self.path, 'test_npynorm_of_ShapeNet/3D index {}.npy'.format(index)))  
            
            norm = torch.tensor(norm, dtype=torch.float32)
            
            coord = (torch.cat((coord, norm),1)).T   
            return coord, norm, one_hot, seg
        else:
            batch = torch.zeros([2048], dtype=torch.int64)
            indexs = fps(coord, batch, ratio=0.03125)
            dss = coord[indexs]
            labelr = self.labels[index]
            index, d_index = self.knn(coord)
            return coord.T, dss.T, coord.T, dss.T, one_hot, seg, index, d_index, indexs

    def __len__(self):
        return self.coordinates.shape[0]

    def output_colored_point_cloud(self, points, seg):
        output_file = 'color_part.ply'
        one = np.ones((points.shape[0], 3))
        for i in range(self.num_points):
            one[i] = self.color_map[seg[i]]
        create_output(points, one*255, output_file)


class Semi_ShapeNetPart(Dataset):
    def __init__(self, args, num_points=2048, ratio=0.0, num_pairs=512,
                 need_norm=True, phase='train', label_rate=0.001):
        self.knn = KNN()
        self.rotate = Rotate()
        self.path = args.data_path
        self.data_path = os.path.join(args.data_path, 'ShapeNet_hdf5_data')
        self.skeleton_path = os.path.join(
            args.data_path, 'ShapeNet_skeletons', phase
        )
        self.skeleton_path = os.path.join(
            args.data_path, 'ShapeNet_skeletons', phase
        )
        self.num_points = num_points
        self.num_classes = 16
        self.num_parts = 50
        self.num_pairs = num_pairs
        self.need_norm = need_norm
        self.phase = phase
        self.classes_dict = {
            'airplane': [0, 1, 2, 3],
            'bag': [4, 5],
            'cap': [6, 7],
            'car': [8, 9, 10, 11],
            'chair': [12, 13, 14, 15],
            'earphone': [16, 17, 18],
            'guitar': [19, 20, 21],
            'knife': [22, 23],
            'lamp': [24, 25, 26, 27],
            'laptop': [28, 29],
            'motorbike': [30, 31, 32, 33, 34, 35],
            'mug': [36, 37],
            'pistol': [38, 39, 40],
            'rocket': [41, 42, 43],
            'skateboard': [44, 45, 46],
            'table': [47, 48, 49]
        }
        self.shape_names = {}
        for cat in self.classes_dict.keys():
            for label in self.classes_dict[cat]:
                self.shape_names[label] = cat

        if phase == 'train':
            files = os.path.join(self.data_path, 'train_hdf5_file_list.txt')
        else:
            files = os.path.join(self.data_path, 'test_hdf5_file_list.txt')
        file_list = [line.rstrip() for line in open(files)]
        num_files = len(file_list)
        self.coordinates = list()
        self.labels = list()
        self.segmentation = list()
        for i in range(num_files):
            cur_file = os.path.join(self.data_path, file_list[i])
            cur_data, cur_label, cur_seg = load_h5_data_label_seg(cur_file) 
            cur_data = cur_data[:, 0:self.num_points, :]
            cur_seg = cur_seg[:, 0:self.num_points]
            self.coordinates.append(cur_data)
            self.labels.append(cur_label)
            self.segmentation.append(cur_seg)  
        self.coordinates = np.vstack(self.coordinates).astype(np.float32)
        self.labels = np.vstack(self.labels).squeeze().astype(np.int64)
        self.segmentation = np.vstack(self.segmentation).astype(np.int64)

        self.category_indices = [[] for _ in range(self.num_classes)]
        for index, l in enumerate(self.labels):
            self.category_indices[l].append(index)

        with open('./data/shape_part_pcindex.txt', 'r') as myfile:
            list_read = (myfile.readlines())
            for i in range(len(list_read)):
                list_read[i] = list_read[i].split(',')[0:-1]
        self.part_indices = list_read
        self.samples =set('0') 
        if phase == 'train':
            self.labeled_sample = []
            for i in range(self.num_parts):
                count = 1
                selected_index = random.sample(self.part_indices[i], count)
                for j in selected_index:
                    self.samples.add(str(j))
        self.labeled_sample = list(self.samples)

        color_map_file = os.path.join(self.data_path, 'part_color_mapping.json')
        self.color_map = json.load(open(color_map_file, 'r'))

    def __getitem__(self, index0):
        index = int(self.labeled_sample[index0])
        coord = self.coordinates[index]  # N*3
        coord = torch.tensor(coord, dtype=torch.float32)
        label = self.labels[index]
        seg = self.segmentation[index]  # N
        one_hot = np.zeros(shape=(16, 1), dtype=np.int64)
        one_hot[label, 0] = 1

        if self.phase == 'train':
            indices = list(range(self.num_points))
            np.random.shuffle(indices)
            coord = coord[indices]
            seg = seg[indices]
        if self.need_norm:
            if self.phase == 'train':
                norm = np.load(os.path.join(self.path, 'train_npynorm_of_ShapeNet/3D index {}.npy'.format(index)))  
            if self.phase == 'test':
                norm = np.load(os.path.join(self.path, 'test_npynorm_of_ShapeNet/3D index {}.npy'.format(index)))  
            
            norm = torch.tensor(norm, dtype=torch.float32)
            coord = torch.cat((coord, norm),1)
            return coord.T, one_hot, seg
        else:
            batch = torch.zeros([2048], dtype=torch.int64)
            indexs = fps(coord, batch, ratio=0.03125)
            dss = coord[indexs]
            labelr = self.labels[index]
            index, d_index = self.knn(coord)
            return coord.T, dss.T, coord.T, dss.T, one_hot, seg, index, d_index, indexs

    def __len__(self):
        return len(self.labeled_sample)

    def output_colored_point_cloud(self, points, seg, output_file):
        with open(output_file, 'w') as handle:
            for i in range(self.num_points):
                color = self.color_map[seg[i]]
                handle.write(
                    'v %f %f %f %f %f %f\n' % (
                        points[0][i], points[1][i], points[2][i],
                        color[0], color[1], color[2]
                    )
                )


class Siamese_ShapeNetPart(Dataset):
    def __init__(self, args, num_points=2048, ratio=0.0, num_pairs=512,
                 need_norm=True, phase='train'):
        self.knn = KNN()
        self.rotate = Rotate()
        self.path = args.data_path
        self.data_path = os.path.join(args.data_path, 'ShapeNet_hdf5_data')
        self.skeleton_path = os.path.join(
            args.data_path, 'ShapeNet_skeletons', phase
        )
        self.skeleton_path = os.path.join(
            args.data_path, 'ShapeNet_skeletons', phase
        )
        self.num_points = num_points
        self.num_classes = 16
        self.num_parts = 50
        self.num_pairs = num_pairs
        self.need_norm = need_norm
        self.phase = phase
        self.classes_dict = {
            'airplane': [0, 1, 2, 3],
            'bag': [4, 5],
            'cap': [6, 7],
            'car': [8, 9, 10, 11],
            'chair': [12, 13, 14, 15],
            'earphone': [16, 17, 18],
            'guitar': [19, 20, 21],
            'knife': [22, 23],
            'lamp': [24, 25, 26, 27],
            'laptop': [28, 29],
            'motorbike': [30, 31, 32, 33, 34, 35],
            'mug': [36, 37],
            'pistol': [38, 39, 40],
            'rocket': [41, 42, 43],
            'skateboard': [44, 45, 46],
            'table': [47, 48, 49]
        }
        self.shape_names = {}
        for cat in self.classes_dict.keys():
            for label in self.classes_dict[cat]:
                self.shape_names[label] = cat

        if phase == 'train':
            files = os.path.join(self.data_path, 'train_hdf5_file_list.txt')
        else:
            files = os.path.join(self.data_path, 'test_hdf5_file_list.txt')
        file_list = [line.rstrip() for line in open(files)]
        num_files = len(file_list) 

        self.coordinates = list()
        self.labels = list()
        self.segmentation = list()
        for i in range(num_files):
            cur_file = os.path.join(self.data_path, file_list[i])
            cur_data, cur_label, cur_seg = load_h5_data_label_seg(cur_file) 
            cur_data = cur_data[:, 0:self.num_points, :]
            cur_seg = cur_seg[:, 0:self.num_points]
            self.coordinates.append(cur_data)
            self.labels.append(cur_label)
            self.segmentation.append(cur_seg)  
        self.coordinates = np.vstack(self.coordinates).astype(np.float32)
        self.labels = np.vstack(self.labels).squeeze().astype(np.int64)
        self.segmentation = np.vstack(self.segmentation).astype(np.int64)

        color_map_file = os.path.join(self.data_path, 'part_color_mapping.json')
        self.color_map = json.load(open(color_map_file, 'r'))

    def __getitem__(self, index):
        coord = self.coordinates[index]  # N*3
        coord = torch.tensor(coord, dtype=torch.float32)
        label = self.labels[index]
        seg = self.segmentation[index]  # N
        one_hot = np.zeros(shape=(16, 1), dtype=np.int64)
        one_hot[label, 0] = 1

        if self.phase == 'train':
            indices = list(range(self.num_points))
            np.random.shuffle(indices)
            coord = coord[indices]
            seg = seg[indices]
        if self.need_norm:
            if self.phase == 'train':
                norm = np.load(os.path.join(self.path, 'train_npynorm_of_ShapeNet/3D index {}.npy'.format(index)))  
            if self.phase == 'test':
                norm = np.load(os.path.join(self.path, 'test_npynorm_of_ShapeNet/3D index {}.npy'.format(index)))  

            norm = torch.tensor(norm, dtype=torch.float32)
            ro_coord = self.rotate(coord, norm)
            batch = torch.zeros([2048], dtype=torch.int64)
            indexs = fps(coord, batch, ratio=0.0078125, random_start=True)
            dss = coord[indexs]
            dn = norm[indexs]
            labelr = self.labels[index]       
            index, d_index = self.knn(coord)
            coord = torch.cat((coord, norm),1)
            return coord.T, ro_coord.T, index, d_index, indexs
        else:
            batch = torch.zeros([2048], dtype=torch.int64)
            indexs = fps(coord, batch, ratio=0.03125)
            dss = coord[indexs]
            labelr = self.labels[index]
            index, d_index = self.knn(coord)
            return coord.T, dss.T, coord.T, dss.T, one_hot, seg, index, d_index, indexs


    def __len__(self):
        return self.coordinates.shape[0]

    def output_colored_point_cloud(self, points, seg):
        output_file = 'color_part.ply'
        one = np.ones((points.shape[0], 3))
        for i in range(self.num_points):
            one[i] = self.color_map[seg[i]]
        create_output(points, one*255, output_file)


class Train_ShapeNetPart(Dataset):
    def __init__(self, args, num_points=2048, ratio=0.0, num_pairs=512,
                 need_norm=True, phase='train'):
        self.knn = KNN()
        self.rotate = Rotate()
        self.path = args.data_path
        self.data_path = os.path.join(args.data_path, 'ShapeNet_hdf5_data')
        self.skeleton_path = os.path.join(
            args.data_path, 'ShapeNet_skeletons', phase
        )
        self.num_points = num_points
        self.num_classes = 16
        self.num_parts = 50
        self.num_pairs = num_pairs
        self.need_norm = need_norm
        self.phase = phase
        self.classes_dict = {
            'airplane': [0, 1, 2, 3],
            'bag': [4, 5],
            'cap': [6, 7],
            'car': [8, 9, 10, 11],
            'chair': [12, 13, 14, 15],
            'earphone': [16, 17, 18],
            'guitar': [19, 20, 21],
            'knife': [22, 23],
            'lamp': [24, 25, 26, 27],
            'laptop': [28, 29],
            'motorbike': [30, 31, 32, 33, 34, 35],
            'mug': [36, 37],
            'pistol': [38, 39, 40],
            'rocket': [41, 42, 43],
            'skateboard': [44, 45, 46],
            'table': [47, 48, 49]
        }
        self.shape_names = {}
        for cat in self.classes_dict.keys():
            for label in self.classes_dict[cat]:
                self.shape_names[label] = cat
        if phase == 'train':
            files = os.path.join(self.data_path, 'train_hdf5_file_list.txt')
        else:
            files = os.path.join(self.data_path, 'test_hdf5_file_list.txt')
        file_list = [line.rstrip() for line in open(files)]
        num_files = len(file_list) 

        self.coordinates = list()
        self.labels = list()
        self.segmentation = list()
        for i in range(num_files):
            cur_file = os.path.join(self.data_path, file_list[i])
            cur_data, cur_label, cur_seg = load_h5_data_label_seg(cur_file) 
            cur_data = cur_data[:, 0:self.num_points, :]
            cur_seg = cur_seg[:, 0:self.num_points]
            self.coordinates.append(cur_data)
            self.labels.append(cur_label)
            self.segmentation.append(cur_seg)  
        self.coordinates = np.vstack(self.coordinates).astype(np.float32)
        self.labels = np.vstack(self.labels).squeeze().astype(np.int64)
        self.segmentation = np.vstack(self.segmentation).astype(np.int64)

        self.siamese_model = Net()
        self.runner = Runner(args)
        self.runner.load_model_weights(self.siamese_model, './log/pretrained_models/siamese_model.pt')
        self.gcn = GCNConv(32, 32)

        color_map_file = os.path.join(self.data_path, 'part_color_mapping.json')
        self.color_map = json.load(open(color_map_file, 'r'))

    def __getitem__(self, index):
        coord = self.coordinates[index]  # N*3
        coord = torch.tensor(coord, dtype=torch.float32)
        label = self.labels[index]
        seg = self.segmentation[index]  # N
        one_hot = np.zeros(shape=(16, 1), dtype=np.int64)
        one_hot[label, 0] = 1

        if self.phase == 'train':
            indices = list(range(self.num_points))
            np.random.shuffle(indices)
            coord = coord[indices]
            seg = seg[indices]
        if self.need_norm:
            if self.phase == 'train':
                norm = np.load(os.path.join(self.path, 'train_npynorm_of_ShapeNet/3D index {}.npy'.format(index)))  
            if self.phase == 'test':
                norm = np.load(os.path.join(self.path, 'test_npynorm_of_ShapeNet/3D index {}.npy'.format(index)))  
            
            norm = torch.tensor(norm, dtype=torch.float32)
            ro_coord = self.rotate(coord, norm)
            batch = torch.zeros([2048], dtype=torch.int64)
            indexs = fps(coord, batch, ratio=0.00390625, random_start=True)   
            labelr = self.labels[index]       
            index, d_index = self.knn(coord)
            coord = (torch.cat((coord, norm),1)).T
            f_simi = ((self.siamese_model(coord.unsqueeze(0))).permute(0, 2, 1)).squeeze(0)       
            fps_f = self.gcn(f_simi, index)             
            
            tmpcnt = 0
            hard_neg = torch.ones(len(indexs),len(indexs))
            for i1 in range(len(indexs)-1):
                for i2 in range(i1+1, len(indexs)):
                    ind1, ind2 = indexs[i1], indexs[i2]
                    embed1, embed2 = fps_f[ind1,:].unsqueeze(0), fps_f[ind2,:].unsqueeze(0)
                    simi = F.cosine_similarity(embed1, embed2)  
                    hard_neg[i1][i2] = simi
                    hard_neg[i2][i1] = simi
            return coord, ro_coord.T, index, d_index, indexs, hard_neg.detach()
        else:
            batch = torch.zeros([2048], dtype=torch.int64)
            indexs = fps(coord, batch, ratio=0.03125)
            dss = coord[indexs]
            labelr = self.labels[index]
            index, d_index = self.knn(coord)
            return coord.T, dss.T, coord.T, dss.T, one_hot, seg, index, d_index, indexs


    def __len__(self):
        return self.coordinates.shape[0]

    def output_colored_point_cloud(self, points, seg):
        output_file = 'color_part.ply'
        one = np.ones((points.shape[0], 3))
        for i in range(self.num_points):
            one[i] = self.color_map[seg[i]]
        create_output(points, one*255, output_file)


class ModelNet40(Dataset):
    def __init__(self, data_path, num_points=1024, transform=None, phase='train'):
        self.data_path = os.path.join(data_path, 'modelnet40_ply_hdf5_2048')
        self.num_points = num_points
        self.num_classes = 40
        self.transform = transform
        self.phase = phase

        # store data
        shape_name_file = os.path.join(self.data_path, 'shape_names.txt')
        self.shape_names = [line.rstrip() for line in open(shape_name_file)]
        self.coordinates = []
        self.labels = []
        self.normals = []
        try:
            files = os.path.join(self.data_path, '{}_files.txt'.format(phase))
            files = [line.rstrip() for line in open(files)]
            for index, file in enumerate(files):
                file_name = file.split('/')[-1]
                files[index] = os.path.join(self.data_path, file_name)
        except FileNotFoundError:
            raise ValueError('Unknown phase or invalid data path.')
        for file in files:
            current_data, current_label, current_normal = load_h5(file)
            current_normal = current_normal[:, 0:self.num_points, :]
            current_data = current_data[:, 0:self.num_points, :]
            self.coordinates.append(current_data)
            self.labels.append(current_label)
            self.normals.append(current_normal)
        self.coordinates = np.vstack(self.coordinates).astype(np.float32)
        self.normals = np.vstack(self.normals).astype(np.float32)
        self.labels = np.vstack(self.labels).squeeze().astype(np.int64)


    def __len__(self):
        if self.phase == 'train':
            return 9840
        else:
            return 2468

    def __getitem__(self, index):
        dats = self.coordinates[index]    
        dats = torch.tensor(dats, dtype=torch.float32)
        normals = self.normals[index]
        normals = torch.tensor(normals, dtype=torch.float32)
        labelr = self.labels[index]       
        return dats, normals, labelr
        

def main():
    print("main is running")
    loader = ShapeNetPart('./data', phase='train') 
    for i in range(200):
        loader.__getitem__(i)

if __name__ == '__main__':
    main()
