#!/usr/bin/env python

import sys
from torch import nn
import torch.nn.functional as F
from utils import weights_init_xavier, weights_init_kaiming, weights_init_normal, weights_init_sparse

class SentimentClassifier(nn.Module):

    def __init__(self, input_dimensionality, num_of_topics, num_of_classes, hid_size, device="cpu" ):

        super(SentimentClassifier, self).__init__()
        
        self.input_dimensionality   = input_dimensionality
        self.num_of_topics          = num_of_topics
        self.num_of_classes         = num_of_classes
        self.hidden_size            = hid_size

        self.frozen = False       
        self.dr     = nn.Dropout(0.2)

        self.fc1    = nn.Linear(self.num_of_topics, self.hidden_size).to(device)
        self.fc2    = nn.Linear(self.hidden_size, self.num_of_classes).to(device)
        self.fc_bn  = nn.BatchNorm1d(self.hidden_size)

        self.device = device

        weights_init_xavier(self.fc1)
        weights_init_xavier(self.fc2)


    def forward(self, z):
        out    = self.fc_bn(F.relu(self.fc1(z)))
        out    = self.fc2(out)
        y_pred = out 
        return y_pred


    def freeze_parameters(self, freeze):
        for layer in [self.fc1, self.fc2]:
            for param in layer.parameters():
                if freeze:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        if freeze:
            self.frozen = True
        else:
            self.frozen = False