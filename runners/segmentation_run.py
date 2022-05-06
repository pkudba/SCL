import os
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
from torch.distributions.normal import Normal

from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader

import sys
sys.path.append("..")
from models.model import Model
from runners.base_run import Runner
from tools.utils import import_class
from models.classifier import Classifier
from data.dataset import ShapeNetPart


class SegmentationRunner(Runner):
    def __init__(self,args):
        super(SegmentationRunner, self).__init__(args)
        # loss
        self.loss = nn.NLLLoss().to(self.output_dev)


    def load_dataset(self,args):
        feeder = ShapeNetPart(args, phase='train') 
        self.print_log("Train data is loading:...")

        train_loader = DataLoader(
                dataset=feeder,
                batch_size=args.train_batch_size*len(args.device),
                shuffle=True,
                num_workers=8 
            )
        
        testfd = ShapeNetPart(args, phase='test') 
        self.print_log("Test data is loading:...")
        test_loader = DataLoader(
                dataset=testfd,
                batch_size=args.test_batch_size*len(args.device),
                shuffle=False,
                num_workers=8 
            )

        self.dataset['train'] = train_loader

        self.shape_names = feeder.shape_names
        self.num_classes = feeder.num_classes
        self.num_parts = feeder.num_parts
        self.num_points = feeder.num_points
        self.print_log(f'Train data loaded: {len(feeder)} samples.')

        self.dataset['test'] = test_loader
        self.print_log(f'Test data loaded: {len(feeder)} samples.')

    def load_model(self, args):
        self.output_dev = self.output_dev
        classifier = Classifier(
            args.detail, num_points=self.num_points, num_classes=self.num_classes,
            num_parts=self.num_parts
        )
        classifier = classifier.to(self.output_dev)
        self.model['train'] = classifier
        backbone = Model(
            k=20, out_features=3
        )
        backbone = backbone.to(self.output_dev)
        self.model['test'] = backbone
        if (args.detail=='4fc-seg' or args.detail=='semi-ft') and self.eval_models==True:
            self.model['test'] = self.model['test'].encoder

    def initialize_model(self, args):
        if args.detail=='4fc-seg':
            if self.eval_models==False:
                self.load_model_weights(self.model['test'],args.backbone_path)
            else:     
                self.load_whole_model(self.model['train'], self.model['test'], './log/pretrained_models/4fc_seg.pt')  
        elif args.detail=='1fc-seg':
            self.load_model_weights(
                self.model['test'],
                './log/pretrained_models/1fc_seg_b.pt'       
            )
            if self.eval_models == True:
                self.load_model_weights(
                    self.model['train'],
                    './log/pretrained_models/1fc_seg_c.pt'
                )
        elif args.detail=='semi-seg':
            self.load_model_weights(
                self.model['test'],
                './log/pretrained_models/semi_seg_b.pt'
            )
            if self.eval_models == True:
                self.load_model_weights(
                    self.model['train'],
                    './log/pretrained_models/semi_seg_c.pt'
                )
        elif args.detail=='semi-ft':
            if self.eval_models==False:
                self.load_model_weights(self.model['test'], args.backbone_path)
            else:     
                self.load_whole_model(self.model['train'], self.model['test'], './log/pretrained_models/semi_ft.pt')       
            
    def run(self, args):
        if self.eval_models==False:
            self.model['test'] = self.model['test'].encoder
        best_epoch = -1
        best_acc = 0.0
        self.epoch = 0
        if self.eval_models == False:
            self.end = 100
        else:
            self.end = 1
        for epoch in range(self.epoch, self.end):
            if self.eval_models == False:
                if self.epoch % 15 == 0 and self.epoch > 0:
                    self.print_log("lr decay...")
                    self.optimizer.param_groups[0]['lr'] *= 0.5
                else:
                    self.print_log("end...")
                self._train_classifier(epoch)
            if self.eval_models == True:
                if args.detail!='4fc-seg' and args.detail!='semi-ft':
                    self.model['test'] = self.model['test'].encoder
                acc = self._eval_classifier(epoch)
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                self.print_log(
                    'Best IoU: {:.2f}%, best model: model{}.pt'.format(
                        best_acc * 100.0, best_epoch + 1
                    ))
            self.epoch += 1

    def _train_classifier(self, epoch):
        self.print_log(f'Train Classifier Epoch: {epoch + 1}')
        tlr = self.optimizer.param_groups[0]['lr']
        self.print_log(f'lr: {tlr}')
        self.model['test'].train()
        self.model['train'].train()

        loader = self.dataset['train']
        loss_values = []

        self.record_time()
        timer = dict(data=0.0, model=0.0, statistic=0.0)
        for batch_id, (x, labels, target) in enumerate(loader):
            # get data
            x = x.float().to(self.output_dev)
            labels = labels.float().to(self.output_dev)
            target = target.long().to(self.output_dev)
            target = target.view(-1, 1)[:, 0]
            timer['data'] += self.tick()

            # forward
            features = self.model['test'](x)
            pred = self.model['train'](features, labels)
            pred = pred.contiguous().view(-1, self.num_parts)
            loss = self.loss(pred, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            timer['model'] += self.tick()

            # statistic
            loss_values.append(loss.item())
            if (batch_id + 1) % 50 == 0:
                self.print_log(
                    'Batch({}/{}) done. Loss: {:.4f}, lr: {:.5f}'.format(
                        batch_id + 1, len(loader), loss.item(),
                        self.optimizer.param_groups[0]['lr']
                    ))
            timer['statistic'] += self.tick()

            if batch_id % 25 == 0:
                self.print_log("batchid {}".format(batch_id))
        
        mean_loss = np.mean(loss_values)
        self.print_log("The epoch {}".format(epoch)+" loss is {}".format(mean_loss))

        self.print_log('Mean training loss: {:.4f}.'.format(mean_loss))
        self.print_log(
            'Time consumption: [Data] {:.1f} min, [Model] {:.1f} min'.format(
                timer['data'] / 60.0, timer['model'] / 60.0
            ))
        if self.eval_models == False:
            model2_path = os.path.join(
                self.classifier_path, f'model{epoch}.pt'
            )

            self.save_whole_weights(
                epoch, self.model['train'], self.model['test'], self.optimizer,
                model2_path
            )

    def _eval_classifier(self, epoch):
        noise_level = 0  # modify the level
        m = Normal(loc=0, scale=noise_level)  
        self.print_log(f'Eval Classifier Epoch: {epoch + 1}')
        self.model['train'].eval()
        self.model['test'].eval()
        self.logs = torch.zeros([16])

        loader = self.dataset['test']
        loss_values = []
        accuracy_values = []
        iou_values = []
        iou_table = np.zeros(shape=(len(self.shape_names), 3))
        with torch.no_grad():
            for batch_id, (x, norm, labels, target) in enumerate(loader):
                x = x.float().to(self.output_dev)
                labels = labels.float().to(self.output_dev)
                target = target.long().to(self.output_dev)

                # forward
                features = self.model['test'](x)
                pred = self.model['train'](features, labels)
        
                # statistic
                iou_table, iou = self._compute_cat_iou(pred, target, iou_table)
                iou_values += iou
                pred = pred.contiguous().view(-1, self.num_parts)
                target = target.view(-1, 1)[:, 0]
                loss = self.loss(pred, target)
                loss_values.append(loss.item())
                pred_indices = pred.data.max(1)[1]
                corrected = pred_indices.eq(target.data).cpu().sum()
                batch_size, _, num_points = x.size()
                accuracy_values.append(
                    corrected.item() / (batch_size * num_points))
                if (batch_id + 1) % 50 == 0:
                    self.print_log(
                        'Batch({}/{}) done. Loss: {:.4f}'.format(
                            batch_id + 1, len(loader), loss.item()
                        ))

        print(loss.values)
        mean_loss = np.mean(loss_values)
        mean_accuracy = np.mean(accuracy_values)
        mean_iou = np.mean(iou_values)
        self.print_log('Mean testing loss: {:.4f}.'.format(mean_loss))
        self.print_log('Mean accuracy: {:.2f}%.'.format(mean_accuracy * 100.0))
        self.print_log('Mean IoU: {:.2f}%.'.format(mean_iou * 100.0))
        return mean_iou

    @staticmethod
    def _compute_cat_iou(pred, target, iou_table):
        iou_list = []
        target = target.cpu().data.numpy()
        for j in range(pred.size(0)):
            batch_pred = pred[j]
            batch_target = target[j]
            batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()
            for cat in np.unique(batch_target):
                intersection = np.sum(
                    np.logical_and(batch_choice == cat, batch_target == cat)
                )
                union = np.sum(
                    np.logical_or(batch_choice == cat, batch_target == cat)
                )
                iou = intersection / float(union) if union != 0 else 1.0
                iou_table[cat, 0] += iou
                iou_table[cat, 1] += 1
                iou_list.append(iou)
        return iou_table, iou_list

def main():
    runner = SegmentationRunner()
    runner.run()


if __name__ == '__main__':
    main() 
    