import os
import torch

import numpy as np
import pandas as pd
import torch.nn as nn

from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from model import Model
from base_exe import Runner
from utils import import_class
from classifier import Classifier
from cla_prepro import ShapeNetPart
# from semi_prepro import ShapeNetPart

train_batch_size = 12
test_batch_size = 12
# device_ids = [0, 1]
# device_ids = [3]
# device_ids = [1, 2]
# device_ids = [1]
# device_ids = [3, 4]
# device_ids = [5, 6]
device_ids = [2, 3]
# device_ids = [4, 5]
# device_ids = [6, 7]
# device_ids = [2, 3]

class ClassifierRunner(Runner):
    def __init__(self):
        super(ClassifierRunner, self).__init__()
        # loss
        self.loss = nn.NLLLoss().to(self.output_dev)

    def load_dataset(self):
        feeder = ShapeNetPart('/data/pkudba/', phase='train') 
        print("Train data is loading:...")
        # train_loader = DataLoader(
        #         dataset=feeder,
        #         batch_size=train_batch_size*len(device_ids),
        #         shuffle=True,
        #         num_workers=8 
        #     )

        train_loader = DataLoader(
                dataset=feeder,
                batch_size=train_batch_size*len(device_ids),
                shuffle=False,
                num_workers=8 
            )
        
        testfd = ShapeNetPart('/data/pkudba/', phase='test') 
        print("Test data is loading:...")
        test_loader = DataLoader(
                dataset=testfd,
                batch_size=test_batch_size*len(device_ids),
                shuffle=True,
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
        print("yes-------------------2")
        print(self.epoch)

    def load_model(self):
        self.output_dev = device_ids[0]
        classifier = Classifier(
            num_points=self.num_points, num_classes=self.num_classes,
            num_parts=self.num_parts
        )
        # classifier = nn.DataParallel(classifier, device_ids=device_ids)
        # classifier = classifier.to(device_ids[0])
        classifier = classifier.to(self.output_dev)
        self.model['train'] = classifier
        backbone = Model(
            k=20, out_features=3
        )
        backbone = backbone.to(device_ids[0])
        self.model['test'] = backbone
        print("yes-------------------3")
        print(self.epoch)

    def initialize_model(self):
        self.load_model_weights(
            self.model['test'], 
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/segmentation/6d_Contras_k80/6.pt'
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/segmentation/6d_Contras_k160_rand/7.pt'

            # '/home/pkudba/BY/from_2020_11_25/Contrastive/segmentation/6d_Contras_k160/5.pt'        # 78%
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model/5.pt'
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model/7.pt'
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal/5.pt'
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_2/16.pt'
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_3_norotate/5.pt'      # cla_model_3
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_3_norotate/10.pt'     # cla_model_3_2
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_3_norotate/14.pt'       # cla_model_3_3
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_3_norotate/16.pt'       # cla_model_3_3
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_3_norotate/19.pt'       # cla_model_3_4     30.pt 71.98
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contrJointly_training/Contras_new4fc_model_6_k40_5per_2as_model_anneal_3_norotate/30.pt'       # cla_model_3_5     30.pt 72.52
            
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_4_norotate/6.pt'        # cla_model_4      30.pt  76.81%
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_4_norotate/8.pt'      # cla_model_4_2         30.pt 77.57
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_4_norotate/10.pt'         # cla_model_4_3   30.pt 77.87

            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_4_norotate/30.pt'       # cla_model_4_4     30.pt 77.98   # Cla_model_4_4_1fc, true  75.2     
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/super_para_experiment/k_80/20.pt'           # super_exp
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/super_para_experiment/k_320/12.pt'           # super_exp
            '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/super_para_experiment/k_20/13.pt'           # super_exp
          
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_4_norotate/58.pt'       # cla_model_4_5       30.pt 77.66
            #  '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_4_norotate/62.pt'       # cla_model_4_6       90.pt 77.97

            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_5_norotate/10.pt'          # 72% Cla_model_5_1
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_5_norotate/15.pt'          # cla_model_5_2

            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_6_norotate_k80/13.pt'       # cla_model_6_1     79.76  Cla_model_6_k80_5per  Cla_new4fc_model_6_k80_5per
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_6_norotate_k80/20.pt'       # cla_model_6_2    79.74
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/sia2_contras_model_anneal_6_norotate_k80/30.pt'    # Cla_new4fc_model_sia2_k80_5per  Cla_model_6_1_sia2  79.66
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_6_norotate_k40/19.pt'       # Cla_new4fc_model_6_k40_5per

            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/contras_model_anneal_6_norotate_k40/40.pt'       # Cla_model_6_1_k40 80.38   Cla_model_6_2_k40  80.8 Cla_model_6_2_k40_1fc_semi
            # Cla_model_6_2_k40_2fc_semi 56%, Cla_model_4fc_semi 49.86 Jointly_training/Contras_new4fc_model_6_k40_5per_2     Cla_model_2fc_semi_2
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/sia2_contras_model_anneal_6_norotate_k40/19.pt'       # Cla_model_6_k40_sia2       79.70

            # '/home/pkudba/BY/from_2020_11_25/Contrastive/segmentation/6d_Contras_k160_ro/8.pt'
            # '/home/pkudba/BY/from_2020_11_25/Contrastive/segmentation/6d_Contras_k160_ro/13.pt'
        )

        # test stage
        if self.eval_models == 1:
            self.load_model_weights(
                self.model['train'],
                # '/home/pkudba/BY/from_2020_11_25/Contrastive/segmentation/6d_Cla_k160_4fc/model{}.pt'.format(11)
                # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/semi/Cla_model_2fc_semi_2/model{}.pt'.format(90)  # 53

                # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/Cla_model_1fc_all/Cla_model_6_2_k40_2fc_semi/model{}.pt'.format(80)
                
                # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/Cla_model_1fc_all/Cla_model_4_4_1fc_true/model{}.pt'.format(30)
                # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/stable/Cla_model_4_4_1fc_true/model{}.pt'.format(40)
                # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/stable/5_Cla_model_4_4_1fc_true/model{}.pt'.format(30)
                # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/super_para_classifiers/k_80_1fc/model{}.pt'.format(50)   # 77.36
                # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/super_para_classifiers/k_320_1fc/model{}.pt'.format(50)      # 72.80
                '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/super_para_classifiers/k_20_1fc/model{}.pt'.format(5)   #  71.98

                # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/Cla_model_1fc_all/Cla_model_6_2_k40_1fc_semi/model{}.pt'.format()
                # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/semi/2fc_part_5per/model{}.pt'.format(90)
                # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/Cla_model_6_2_k40/model{}.pt'.format(85)
                # '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/semi/1fc_part_single/model{}.pt'.format(90)
                
            )
            # self.load_optimizer_weights(self.optimizer, '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/Cla_model_1fc_all/Cla_model_4_4_1fc_true/model{}.pt'.format(30))
            # self.load_scheduler_weights(self.scheduler, '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/Cla_model_1fc_all/Cla_model_4_4_1fc_true/model{}.pt'.format(30))

            # self.load_optimizer_weights(self.optimizer[0], '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/Cla_model_5_1_percent_semi/Cla_new4fc_model_6_k40_5per/model{}.pt'.format(30))
            # self.load_scheduler_weights(self.scheduler, '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/Cla_model_5_1_percent_semi/Cla_new4fc_model_6_k40_5per/model{}.pt'.format(30))
            
            # self.load_optimizer_weights(self.optimizer, '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/Cla_model_6_2/model{}.pt'.format(36))
            # self.load_scheduler_weights(self.scheduler, '/home/pkudba/BY/from_2020_11_25/Contrastive/Siamese/Cla_model_6_2/model{}.pt'.format(36))
        print("yes-------------------6")
        print(self.epoch)

    def run(self):
        # print(type(self.optimizer.param_groups[0]))
        # exit()
        best_epoch = -1
        best_acc = 0.0
        self.model['test'] = self.model['test'].encoder
        self.epoch = 0
        if self.eval_models == 0:
            self.end = 100
        else:
            self.end = 1
        for epoch in range(self.epoch, self.end):
            print("yes-------------------1")
            print(self.epoch)
            if self.eval_models == 0:
                print("-------------------------")
                if self.epoch % 15 == 0 and self.epoch > 0:
                    print("lr decay...")
                    self.optimizer.param_groups[0]['lr'] *= 0.5
                else:
                    print("end...")
                self._train_classifier(epoch)
            '''
            eval_model = self.args.eval_model and (
                    ((epoch + 1) % self.args.eval_interval == 0) or
                    (epoch + 1 == self.args.num_classifier_epochs))
            '''
            if self.eval_models == 1:
                print("********************")
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
        self.model['test'].eval()
        self.model['train'].train()

        loader = self.dataset['train']
        # print(type(loader))
        # exit()
        loss_values = []

        self.record_time()
        timer = dict(data=0.0, model=0.0, statistic=0.0)
        # for batch_id, (x, ds, norm, labels, target, index, d_index) in enumerate(loader):
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
                print("batchid {}".format(batch_id))
        # self.scheduler.step()
        
        mean_loss = np.mean(loss_values)
        print("------------------------------------")
        print("The epoch {}".format(epoch)+" loss is {}".format(mean_loss))

        self.print_log('Mean training loss: {:.4f}.'.format(mean_loss))
        self.print_log(
            'Time consumption: [Data] {:.1f} min, [Model] {:.1f} min'.format(
                timer['data'] / 60.0, timer['model'] / 60.0
            ))
        if self.eval_models == 0:
            model_path = os.path.join(
                self.classifier_path, f'model{epoch}.pt'
            )
            self.save_weights(
                epoch, self.model['train'], self.optimizer, model_path
            )

            # model2_path = os.path.join(
            #     self.new_contras_path, f'model{epoch}.pt'
            # )

            # self.save_weights(
            #     epoch, self.model['train'], self.model['test'], self.optimizer, model2_path
            # )
        '''
        if self.args.use_tensorboard:
            with SummaryWriter(log_dir=self.tensorboard_path) as writer:
                writer.add_scalar('train/classifier_loss', mean_loss, epoch)
        '''

    def _eval_classifier(self, epoch):
        self.print_log(f'Eval Classifier Epoch: {epoch + 1}')
        self.model['train'].eval()
        self.model['test'].eval()

        loader = self.dataset['test']
        loss_values = []
        accuracy_values = []
        iou_values = []
        iou_table = np.zeros(shape=(len(self.shape_names), 3))
        with torch.no_grad():
            for batch_id, (x, labels, target) in enumerate(loader):
                # print(labels.shape, type(labels),target.shape, type(target))  # torch.Size([1, 16, 1]) <class 'torch.Tensor'> torch.Size([1, 2048]) <class 'torch.Tensor'>
                # exit()
                # get data
                # print(x.shape)
                # exit()
                x = x.float().to(self.output_dev)
                labels = labels.float().to(self.output_dev)
                target = target.long().to(self.output_dev)

                # forward
                features = self.model['test'](x)
                pred = self.model['train'](features, labels)
                # print(pred, target)
                # exit()

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

        print("yes-------------------8")
        print(self.epoch)
        mean_loss = np.mean(loss_values)
        mean_accuracy = np.mean(accuracy_values)
        mean_iou = np.mean(iou_values)
        self.print_log('Mean testing loss: {:.4f}.'.format(mean_loss))
        self.print_log('Mean accuracy: {:.2f}%.'.format(mean_accuracy * 100.0))
        self.print_log('Mean IoU: {:.2f}%.'.format(mean_iou * 100.0))

        # print((iou_table[0][0]+iou_table[1][0]+iou_table[2][0]+iou_table[3][0])/(iou_table[0][1]+iou_table[1][1]+iou_table[2][1]+iou_table[3][1]))
        # print((iou_table[4][0]+iou_table[5][0])/(iou_table[4][1]+iou_table[5][1]))
        # print((iou_table[6][0]+iou_table[7][0])/(iou_table[6][1]+iou_table[7][1]))
        # print((iou_table[8][0]+iou_table[9][0]+iou_table[10][0]+iou_table[11][0])/(iou_table[8][1]+iou_table[9][1]+iou_table[10][1]+iou_table[11][1]))
        # print((iou_table[12][0]+iou_table[13][0]+iou_table[14][0]+iou_table[15][0])/(iou_table[12][1]+iou_table[13][1]+iou_table[14][1]+iou_table[15][1]))        
        # print((iou_table[16][0]+iou_table[17][0]+iou_table[18][0])/(iou_table[16][1]+iou_table[17][1]+iou_table[18][1]))
        # print((iou_table[19][0]+iou_table[20][0]+iou_table[21][0])/(iou_table[19][1]+iou_table[20][1]+iou_table[21][1]))
        # print((iou_table[22][0]+iou_table[23][0])/(iou_table[22][1]+iou_table[23][1]))
        # print((iou_table[24][0]+iou_table[25][0]+iou_table[26][0]+iou_table[27][0])/(iou_table[24][1]+iou_table[25][1]+iou_table[26][1]+iou_table[27][1]))
        # print((iou_table[28][0]+iou_table[29][0])/(iou_table[28][1]+iou_table[29][1]))
        # print((iou_table[30][0]+iou_table[31][0]+iou_table[32][0]+iou_table[33][0]+iou_table[34][0]+iou_table[35][0])/(iou_table[30][1]+iou_table[31][1]+iou_table[32][1]+iou_table[33][1]+iou_table[34][1]+iou_table[35][1]))
        # print((iou_table[36][0]+iou_table[37][0])/(iou_table[36][1]+iou_table[37][1]))
        # print((iou_table[38][0]+iou_table[39][0]+iou_table[40][0])/(iou_table[38][1]+iou_table[39][1]+iou_table[40][1]))
        # print((iou_table[41][0]+iou_table[42][0]+iou_table[43][0])/(iou_table[41][1]+iou_table[42][1]+iou_table[43][1]))
        # print((iou_table[44][0]+iou_table[45][0]+iou_table[46][0])/(iou_table[44][1]+iou_table[45][1]+iou_table[46][1]))
        # print((iou_table[47][0]+iou_table[48][0]+iou_table[49][0])/(iou_table[47][1]+iou_table[48][1]+iou_table[49][1]))
        # print(iou_table)
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
    runner = ClassifierRunner()
    runner.run()


if __name__ == '__main__':
    main() 