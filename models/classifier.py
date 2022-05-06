import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import sys
sys.path.append("..")
from tools.utils import get_total_parameters

class Classifier(nn.Module):
    def __init__(self, detail, in_features=256, num_points=2048, num_classes=16,
                 num_parts=50):
        super(Classifier, self).__init__()
        self.num_points = num_points
        if detail=='4fc-seg':
            self.num_fc = 4
        elif detail=="1fc-seg":
            self.num_fc = 1
        else:
            self.num_fc = 2
        self.conv0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(in_features, 1024, kernel_size=1, bias=False)),
            ('bn0', nn.BatchNorm1d(1024)),
            ('relu0', nn.LeakyReLU(negative_slope=0.2))
        ]))
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(num_classes, 64, kernel_size=1, bias=False)),
            ('bn0', nn.BatchNorm1d(64)),
            ('relu0', nn.LeakyReLU(negative_slope=0.2))
        ]))
        if self.num_fc == 1:
            self.classifier = (nn.Conv1d(576, 50, kernel_size=1, bias=True))
        # the classifier of 78% miou
        elif self.num_fc == 2:
            self.classifier = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv1d(576, 256, kernel_size=1, bias=False)),
                ('bn0', nn.BatchNorm1d(256)),
                ('relu0', nn.LeakyReLU(negative_slope=0.2)),
                ('drop0', nn.Dropout(p=0.4)),
                ('conv1', nn.Conv1d(256, 50, kernel_size=1, bias=False)),
            ]))
        elif self.num_fc == 3:
            self.classifier = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv1d(576, 256, kernel_size=1, bias=False)),
                ('bn0', nn.BatchNorm1d(256)),
                ('relu0', nn.LeakyReLU(negative_slope=0.2)),
                ('drop0', nn.Dropout(p=0.5)),
                ('conv1', nn.Conv1d(256, 128, kernel_size=1, bias=False)),
                ('bn1', nn.BatchNorm1d(128)),
                ('relu1', nn.LeakyReLU(negative_slope=0.2)),
                ('drop1', nn.Dropout(p=0.5)),
                ('conv2', nn.Conv1d(128, 50, kernel_size=1, bias=False)),
            ]))
        else:
            self.classifier = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv1d(1344, 2048, kernel_size=1, bias=False)),
                ('bn0', nn.BatchNorm1d(2048)),
                ('relu0', nn.LeakyReLU(negative_slope=0.2)),
                ('drop0', nn.Dropout(p=0.4)),
                ('conv1', nn.Conv1d(2048, 4096, kernel_size=1, bias=False)),
                ('bn1', nn.BatchNorm1d(4096)),
                ('relu1', nn.LeakyReLU(negative_slope=0.2)),  
                ('drop1', nn.Dropout(p=0.4)),
                ('conv2', nn.Conv1d(4096, 1024, kernel_size=1, bias=False)),
                ('bn2', nn.BatchNorm1d(1024)),
                ('relu2', nn.LeakyReLU(negative_slope=0.2)),
                ('conv3', nn.Conv1d(1024, 50, kernel_size=1, bias=False)),
            ]))

    def forward(self, x, labels):
        if self.num_fc == 4:
            features = self.conv0(x)
            features = self.max_pool(features)
            labels = self.conv1(labels)
            features = torch.cat((features, labels), dim=1)
            features = features.repeat(1, 1, self.num_points)
            features = torch.cat((features, x), dim=1)
            features = self.classifier(features)
            features = features.permute(0, 2, 1).contiguous()
            features = F.log_softmax(features, dim=-1)
        else:
            features = self.max_pool(x)
            labels = self.conv1(labels)
            features = torch.cat((features, labels), dim=1)
            features = features.repeat(1, 1, self.num_points)
            features = torch.cat((features, x), dim=1)
            features = self.classifier(features)
            features = features.permute(0, 2, 1).contiguous()
            features = F.log_softmax(features, dim=-1)
        return features


def main():
    labels = torch.rand(4, 16, 1)
    features = torch.rand(4, 256, 2048)
    classifier = Classifier()
    print('Classifier:', get_total_parameters(classifier))
    score = classifier(features, labels)



if __name__ == '__main__':
    main()
