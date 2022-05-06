import collections
import json
import os
import random
import time
import torch
import yaml

import numpy as np
import torch.nn as nn

from abc import abstractmethod

class Runner:
    def __init__(self, args):
        # register variables
        self.backbone_path = None
        self.classifier_path = None
        self.tensorboard_path = None
        self.dataset = {'train': None, 'test': None}
        self.model = {'train': None, 'test': None}
        self.optimizer = None
        self.scheduler = None

        self.cur_time = 0
        self.epoch = 0

        if args.use_cuda:
            if type(args.device) is list:
                self.device = args.device[0]
            else:
                self.device = args.device
        else:
            self.device = 'cpu'

        self.load_dataset(args)
        self.load_model()
        self.initialize_model()
        
    
    @abstractmethod
    def load_dataset(self, args):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def initialize_model(self):
        pass
    
    def load_optimizer(self):
        if self.model['train'] is None:
            return
        if 'sgd' in self.args.optimizer.lower():
            paras = self.model['train'].parameters()
            try:
                self.optimizer = torch.optim.SGD(
                    paras,
                    lr=self.args.lr,
                    momentum=self.args.momentum,
                    weight_decay=1e-4
                )
            except ValueError as e:
                self.print_log(str(e))
        elif 'adam' in self.args.optimizer.lower():
            paras = self.model['train'].parameters()
            try:
                self.optimizer = torch.optim.Adam(
                    paras,
                    lr=self.args.lr,
                    weight_decay=1e-4
                )
            except ValueError as e:
                self.print_log(str(e))
        else:
            raise ValueError('Unsupported optimizer.')
    
    @abstractmethod
    def run(self):
        pass

    def load_whole_model(self, model2, weights_file, ignore=None):
        self.print_log(f'Loading model weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        model2_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.to(self.device))
            for k, v in check_points['model2'].items()
        ])
        self._try_load_weights(model2, model2_weights)
    

    def load_model_weights(self, model, weights_file, ignore=[]):
        self.print_log(f'Loading model weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        self.epoch = check_points['epoch'] + 1
        model_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.to(self.device))
            for k, v in check_points['model'].items()
        ])
        self._try_load_weights(model, model_weights)
   
    def _try_load_weights(self, model, weights):
        try:
            model.load_state_dict(weights)
        except:
            state = model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            self.print_log('Can not find these weights:')
            for d in diff:
                self.print_log(d)
            state.update(weights)
            model.load_state_dict(state)
    
    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def tick(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, msg, print_time=True):
        if print_time:
            localtime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
            msg = "[" + localtime + '] ' + msg
        print(msg)
    