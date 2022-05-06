import torch

import numpy as np
from torch.distributions.normal import Normal
from torch_geometric.nn import fps


from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from torch.utils.data.dataloader import DataLoader
from sklearn import preprocessing

import sys
sys.path.append("..")
from models.transfer_layer import Pooler
from models.transfer_model import Model
from runners.cla_run import Runner
from data.dataset import ModelNet40

from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np


class ClassifierRunner(Runner):
    def __init__(self,args):
        super(ClassifierRunner, self).__init__(args)
        self.device = self.device

    def load_dataset(self, args):
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        feeder = ModelNet40(args.data_path, phase='train')
        self.print_log("Train data is loading:...")
        train_data = DataLoader(
                dataset=feeder,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=8
            )
        self.dataset['train'] = train_data
        self.shape_names = feeder.shape_names
        self.print_log(f'Train data loaded: {len(feeder)} samples.')
        
        testfd = ModelNet40(args.data_path, phase='test')
        self.print_log("Test data is loading:...")
        test_data = DataLoader(
                dataset=testfd,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=8 
            )
        self.dataset['test'] = test_data
        self.print_log(f'Test data loaded: {len(feeder)} samples.')

    def load_model(self):
        if torch.cuda.is_available():
            self.print_log("CUDA is Running...")
        else:
            self.print_log("CUDA is not Running and the cpu is Running...")
        md = Model()
        md.to(self.device)

        pooler = Pooler()
        pooler = pooler.to(self.device)
        self.model['train'] = pooler
        self.model['test'] = md
        self.classifier = OneVsRestClassifier(
            LinearSVC(C=1300.0, max_iter=10240, dual=False, intercept_scaling=8)
        )

    def initialize_model(self):
        self.model['test'] = self.model['test'].encoder
        self.load_whole_model(self.model['test'], './log/pretrained_models/classifier_b.pt')
        self.epoch = 0
    

    def run(self, args):
        best_epoch = -1
        best_acc = 0.0
        for epoch in range(self.epoch, 1):
            self._train_classifier(epoch)
            self.print_log("--------------------------train_classifier runner ended----------------------------")
            acc = self._eval_classifier(epoch)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
            self.print_log('Best accuracy: {:.2f}%, best model: model{}.pt'.format(best_acc * 100.0, best_epoch + 1))


    def _train_classifier(self, epoch):
        self.print_log('Train Classifier Epoch: {}'.format(epoch + 1))
        self.model['test'].eval()
        self.model['train'].train()

        loader = self.dataset['train']

        self.record_time()
        batch_features = []
        batch_labels = []
        timer = dict(data=0.0, model=0.0, statistic=0.0)
        with torch.no_grad():
            for batch_id, (dat, normals, label) in enumerate(loader):
                dat = (torch.cat((dat, normals),dim=-1)).permute(0, 2, 1)
                dat = dat.to(self.device)   
                features = self.model['test'](dat)
                label = label.long().to(self.device)
                features = self.model['train'](features)
                features = preprocessing.normalize(features.cpu(), norm='l2')
                features = torch.from_numpy(features)
                if (batch_id + 1) % 50 == 0:
                    self.print_log('Batch({}/{}) done.'.format(
                        batch_id + 1, len(loader)
                    ))
                batch_features.append(features.cpu().numpy())
                batch_labels.append(label.cpu().numpy().reshape((-1, 1)))

            self.print_log("-----train_classifier_batch over-------")

        timer['data'] += self.tick()
        batch_features = np.vstack(batch_features)
        batch_labels = np.vstack(batch_labels).reshape((-1,)).astype(int)
        self.print_log("-----np.vstack over-------")

        self.classifier.fit(batch_features, batch_labels)
        timer['model'] += self.tick()
        self.print_log("------over--------")
        '''
        self.print_log(
            'Time consumption: [Data] {:.1f} min, [Model] {:.1f} min'.format(
                timer['data'] / 60.0, timer['model'] / 60.0
            ))
        self.print_log("------print_log-----over")
        '''


    def _eval_classifier(self, epoch):
        noise_level = 0  # modify the level
        m = Normal(loc=0, scale=noise_level)  
        self.print_log('Eval Classifier Epoch: {}'.format(epoch + 1))
        self.model['test'].eval()
        self.print_log("eval begin...")
        loader = self.dataset['test']
        pred_scores = []
        true_scores = []
        Pred_all = []
        Label_all = []
        with torch.no_grad():
            for batch_id, (dat, normals, label) in enumerate(loader):
                if (batch_id + 1) % 50 == 0:
                    self.print_log('Batch({}/{}) begin.'.format(
                        batch_id + 1, len(loader)
                    ))
                '''
                density=512
                den_dat = torch.zeros([self.test_batch_size, density, 3])
                den_normals = torch.zeros([self.test_batch_size, density, 3])
                my_ratio = 0.5

                batch = torch.zeros([1024], dtype=torch.int64)
                for i in range(dat.shape[0]):
                    # new_ind = fps(dat[i], batch, ratio=0.25, random_start=True)
                    # new_ind = fps(dat[i], batch, ratio=0.5, random_start=True)
                    new_ind = fps(dat[i], batch, ratio=my_ratio, random_start=True)
                    den_dat[i] = dat[i][new_ind]
                    den_normals[i] = normals[i][new_ind]
                # simple adjusting add density ablation study
                #dat = (torch.cat((den_dat, den_normals),dim=-1)).permute(0, 2, 1)
                '''
                dat = (torch.cat((dat, normals),dim=-1)).permute(0, 2, 1) 
                dat = dat.to(self.device)

                features = self.model['test'](dat)
                label = label.long().to(self.device)
                features = self.model['train'](features)
                features = preprocessing.normalize(features.cpu(), norm='l2')
                features = torch.from_numpy(features)

                current_features = features.detach().cpu().numpy()
                current_labels = label.cpu().numpy()
                label = label.detach().cpu().numpy()

                pred = self.classifier.predict(current_features)
                for i in range(pred.shape[0]):
                    Pred_all.append(pred[i])
                    Label_all.append(label[i])
        
                pred_scores.append(pred)
                true_scores.append(current_labels)
        self.print_log(len(Pred_all), len(Label_all), max(Pred_all), min(Pred_all))

        pred_scores = np.concatenate(pred_scores)
        true_scores = np.concatenate(true_scores)
        overall_acc = accuracy_score(true_scores, pred_scores)
        avg_class_acc = balanced_accuracy_score(true_scores, pred_scores)
        self.print_log('Overall accuracy: {:.2f}%'.format(overall_acc * 100.0))
        self.print_log(
            'Average class accuracy: {:.2f}%'.format(avg_class_acc * 100.0)
        )
        return overall_acc


def main():
    runner = ClassifierRunner()
    runner.run()


if __name__ == '__main__':
    main() 
