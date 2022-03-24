# General torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Imports for pretrained encoders
from torchvision import models

# Import for Representation learners
from byol_pytorch import BYOL

# State based Behavior Cloning
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(15, 128)
        self.fc_2 = nn.Linear(128, 512)
        self.fc_3 = nn.Linear(512, 512)
        self.fc_4 = nn.Linear(512, 128)
        self.fc_5 = nn.Linear(128, 12)
        self.batch_norm = nn.BatchNorm1d(512)

    def forward(self, x):
        x = x.view(-1, 15)
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        x = self.fc_3(x)
        x = F.leaky_relu(self.batch_norm(x))
        x = F.leaky_relu(self.fc_4(x))
        x = self.fc_5(x)
        return x

# Visual Behavior Cloning
class BehaviorCloning(nn.Module):
    def __init__(self):
        super(BehaviorCloning, self).__init__()
        # Encoder
        self.encoder = models.resnet50(pretrained = True)
        # Fully Connected Regression Layer
        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, 1024)
        self.fc3 = nn.Linear(1024, 12)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        return x
