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
    def __init__(self,args):
        # register variables
        self.backbone_path = None
        self.classifier_path = None
        self.tensorboard_path = None
        self.cloud_path = None

        self.dataset = {'train': None, 'test': None}
        self.model = {'train': None, 'test': None}
        self.optimizer = None
        self.scheduler = None
        self.scheduler_train = None
        self.scheduler_test = None

        self.cur_time = 0
        self.epoch = 0
        self.eval_models = args.eval_classifier
        self.args = self.check_args(args)
        args_json = json.dumps(vars(self.args), sort_keys=True, indent=2)
        self.print_log(args_json, print_time=False)
        
        # random seed
        if self.args.use_seed:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if self.args.use_cuda:
                torch.cuda.manual_seed_all(args.seed)
        # devices
        if self.args.use_cuda:
            if type(args.device) is list:
                self.output_dev = args.device[0]
            else:
                self.output_dev = args.device
        else:
            self.output_dev = 'cpu'
        
        # model
        self.load_dataset(args)
        self.load_model(args)
        self.load_optimizer()
        self.initialize_model(args)
        
        # data parallel
        if self.eval_models==False or self.args.mode=='train': 
            if type(self.args.device) is list and \
                    len(self.args.device) > 1 and self.args.use_cuda:
                for key, val in self.model.items():
                    if val is None:
                        continue
                    self.model[key] = nn.DataParallel(
                        val, device_ids=args.device, output_device=self.output_dev
                    )
        

    def check_args(self, args):
        self.backbone_path = os.path.join(args.save_dir, 'backbone')
        if not os.path.exists(self.backbone_path):
            os.makedirs(self.backbone_path)
        self.siamese_path = os.path.join(args.save_dir, 'siamese')
        if not os.path.exists(self.siamese_path):
            os.makedirs(self.siamese_path)
        self.classifier_path = os.path.join(args.save_dir, 'classifier')
        if not os.path.exists(self.classifier_path):
            os.makedirs(self.classifier_path)
        self.tensorboard_path = os.path.join(args.save_dir, 'tensorboard')
        if not os.path.exists(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)
        self.cloud_path = os.path.join(args.save_dir, 'point_clouds')
        if not os.path.exists(self.cloud_path):
            os.makedirs(self.cloud_path)

        args.use_cuda = args.use_cuda and torch.cuda.is_available()

        args.num_points = max(1, args.num_points)
        args.knn = max(1, args.knn)

        args.save_interval = max(1, args.save_interval)
        args.eval_interval = max(1, args.eval_interval)
        args.log_interval = max(1, args.log_interval)

        args.train_batch_size = max(1, args.train_batch_size)
        args.test_batch_size = max(1, args.test_batch_size)
        args.num_epochs = max(1, args.num_epochs)

        # save configuration file
        config_file = os.path.join(args.save_dir, 'config.yaml')
        args_dict = vars(args)
        with open(config_file, 'w') as f:
            yaml.dump(args_dict, f)
        return args
    

    @abstractmethod
    def load_dataset(self, args):
        pass

    @abstractmethod
    def load_model(self, args):
        pass

    @abstractmethod
    def initialize_model(self, args):
        pass

    def load_optimizer(self):
        if self.model['train'] is None:
            return
        self.optimizer = torch.optim.Adam(
                self.model['train'].parameters(),
                lr=0.0005, 
                weight_decay=0)
    '''
    def load_scheduler(self):
        # TODO: to support more schedulers
        if self.model['train'] is None:
            return
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 30, eta_min=0.0005 / 100.0
        )
    '''
    @abstractmethod
    def run(self):
        pass

    def load_model_weights(self, model, weights_file, ignore=None):
        self.print_log(f'Loading model weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        # load model weights
        model_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.to(self.output_dev))
            for k, v in check_points['model'].items()
        ])
        self._try_load_weights(model, model_weights)
    
    def load_whole_model(self, model1, model2, weights_file, ignore=None):
        self.print_log(f'Loading model weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        # load model weights
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
        # load optimizer configuration
        optim_weights = check_points['optimizer']
        self._try_load_weights(optimizer, optim_weights)

    def load_scheduler_weights(self, scheduler, weights_file):
        self.print_log(f'Loading scheduler weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        # load scheduler configuration
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

        
    def save_whole_weights(self, epoch, model1, model2, optimizer, save_path):
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
        if self.args.print_log:
            with open(os.path.join(self.args.save_dir, 'log.txt'), 'a') as f:
                print(msg, file=f)
