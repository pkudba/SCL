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

import utils


class Runner:
    def __init__(self):
        # register variables
        self.backbone_path = None
        self.classifier_path = None
        self.tensorboard_path = None
        self.cloud_path = None

        self.transform = None
        self.dataset = {'train': None, 'test': None}
        self.model = {'train': None, 'test': None}
        self.optimizer = None
        self.scheduler = None
        self.scheduler_train = None
        self.scheduler_test = None

        self.cur_time = 0
        self.epoch = 0
        self.eval_models = 1
        self.output_dev = 0
        self.load_dataset()
        self.load_model()
        self.load_optimizer()
        self.initialize_model()
        
        self.new_contras_path = './models/J_3fc_part_5per'
        self.classifier_path = './classifiers/k_20_1fc'

    @abstractmethod
    def load_dataset(self):
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
        self.optimizer = torch.optim.Adam(
                self.model['train'].parameters(),
                # lr=0.0003, 
                lr=0.0005, 
                weight_decay=0)
        print("yes-------------------4")
        print(self.epoch)

    def load_scheduler(self):
        if self.model['train'] is None:
            return
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 30, eta_min=0.0005 / 100.0
        )
        
        print("yes-------------------5")
        print(self.epoch)

    @abstractmethod
    def run(self):
        pass

    def load_model_weights(self, model, weights_file, ignore=None):
        self.print_log(f'Loading model weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        model_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.to(self.output_dev))
            for k, v in check_points['model'].items()
        ])
        self._try_load_weights(model, model_weights)
    
    def Jointly_load_model_weights(self, model1, model2, weights_file, ignore=None):
        self.print_log(f'Loading model weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        model1_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.to(self.output_dev))
            for k, v in check_points['model1'].items()
        ])
        model2_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.to(self.output_dev))
            for k, v in check_points['model2'].items()
        ])
        self._try_load_weights(model1, model1_weights)
        self._try_load_weights(model2, model2_weights)


    def load_optimizer_weights(self, optimizer, weights_file):
        self.print_log(f'Loading optimizer weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        optim_weights = check_points['optimizer']
        self._try_load_weights(optimizer, optim_weights)

    def load_scheduler_weights(self, scheduler, weights_file):
        self.print_log(f'Loading scheduler weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        sched_weights = check_points['scheduler']
        self._try_load_weights(scheduler, sched_weights)

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

    def save_weights_2(self, epoch, model1, model2, optimizer, save_path):
        model1_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.cpu())
            for k, v in model1.state_dict().items()
        ])
        model2_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.cpu())
            for k, v in model2.state_dict().items()
        ])
        save_dict = {
            'epoch': epoch,
            'model1': model1_weights,
            'model2': model2_weights
        }
        torch.save(save_dict, save_path)
        self.print_log('Model ' + save_path + ' saved.')

    
    def save_weights(self, epoch, model, optimizer, save_path):
        model_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.cpu())
            for k, v in model.state_dict().items()
        ])
        save_dict = {
            'epoch': epoch,
            'model': model_weights
        }
        torch.save(save_dict, save_path)
        self.print_log('Model ' + save_path + ' saved.')

        
    def Jointly_save_weights(self, epoch, model1, model2, optimizer, save_path):
        model1_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.cpu())
            for k, v in model1.state_dict().items()
        ])
        model2_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.cpu())
            for k, v in model2.state_dict().items()
        ])
        save_dict = {
            'epoch': epoch,
            'model1': model1_weights,
            'model2': model2_weights
        }
        torch.save(save_dict, save_path)
        self.print_log('Model ' + save_path + ' saved.')


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
