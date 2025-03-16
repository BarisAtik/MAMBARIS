from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
import math
from typing import Union

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional block - increased capacity from 32 to 48 filters
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        
        # Second convolutional block - increased capacity from 64 to 128 filters
        self.conv2 = nn.Conv2d(48, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Global average pooling and final dense layer (updated input features to 128)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Global average pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification layer
        logits = self.fc(x)
        probabilities = F.softmax(logits, dim=-1)
        
        return logits, probabilities