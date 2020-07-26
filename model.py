import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

class BasicBloack(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBloack, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size= 1, stride= 1, padding= 0, bias= False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, in_planes, kernel_size= 3, stride= 1, padding= 1, bias= False)
        self.bn2 = nn.BatchNorm2d(in_planes)
    
    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = F.leaky_relu(out)
        return out

class DownSample(nn.Module):
    def __init__(self, in_planes, planes):
        super(DownSample, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size= 3, stride= 2, padding= 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, in_planes, kernel_size= 1, stride= 1, padding= 0, bias = False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, planes, kernel_size= 3, stride= 1, padding= 1, bias= False)
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out_1 = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.leaky_relu(self.bn2(self.conv2(out_1)))
        out = self.bn3(self.conv3(out))
        out += out_1
        out = F.leaky_relu(out)
        return out

class PAE(nn.Module):

    def __init__(self, img_size = 640):
        super(PAE, self).__init__()
        
