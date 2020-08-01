import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from scripts.utils import *

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        
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
        
        self.img_size = 640
        self.ignore_thres = 0.5
        self.grid_size = 5
        self.obj_scale = 5
        self.noobj_scale = 1
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        # 0
        self.conv1 = nn.Conv2d(3, 32, kernel_size= 3, stride= 1, padding= 1, bias= False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = self._add_DownSample_layer(32, 64, 2) # 1 ~ 8
        self.layer2 = self._add_BasicBlock_layer(128, 64, 1) # 9 ~ 11
        self.layer3 = self._add_DownSample_layer(128, 256, 1) # 12 ~ 15
        self.layer4 = self._add_BasicBlock_layer(256, 128, 7) # 16 ~ 36
        self.layer5 = self._add_DownSample_layer(256, 512, 1) # 37 ~ 40
        self.layer6 = self._add_BasicBlock_layer(512, 256, 4) # 41 ~ 52
        self.layer7 = self._add_DownSample_layer(512, 512, 1) # 53 ~ 56
        self.layer8 = self._add_BasicBlock_layer(512, 256, 3) # 57 ~ 65
        self.layer9 = self._add_DownSample_layer(512, 512, 1) # 66 ~ 69
        self.layer10 = self._add_BasicBlock_layer(512, 256, 2) # 70 ~ 75
        self.layer11 = self._add_DownSample_layer(512, 512, 1) # 76 ~ 79
        self.layer12 = self._add_BasicBlock_layer(512, 256, 2) # 80 ~ 85

        # 86
        self.conv2 = nn.Conv2d(512, 256, kernel_size= 1, stride =1 , padding= 0, bias= False)
        self.bn2 = nn.BatchNorm2d(256)
        # 87
        self.conv3 = nn.Conv2d(256, 512, kernel_size= 3, stride =1 , padding= 1, bias= False)
        self.bn3 = nn.BatchNorm2d(512)
        # 88
        self.conv4 = nn.Conv2d(512, 256, kernel_size= 1, stride =1 , padding= 0, bias= False)
        self.bn4 = nn.BatchNorm2d(256)
        # 89
        self.conv5 = nn.Conv2d(256, 512, kernel_size= 3, stride =1 , padding= 1, bias= False)
        self.bn5 = nn.BatchNorm2d(512)
        # 90
        self.conv6 = nn.Conv2d(512, 256, kernel_size= 1, stride =1 , padding= 0, bias= False)
        self.bn6 = nn.BatchNorm2d(256)
        # 91
        self.conv7 = nn.Conv2d(256, 512, kernel_size= 3, stride =1 , padding= 1, bias= False)
        self.bn7 = nn.BatchNorm2d(512)
        # 92
        self.conv8 = nn.Conv2d(512, 21, kernel_size= 1, stride= 1, padding= 0, bias= False)
        self.bn8 = nn.BatchNorm2d(21)
    
    def _add_DownSample_layer(self, in_planes, planes, num):
        layers = []
        for i in range(num):
            layers.append(DownSample(in_planes, planes))
            in_planes = planes
            planes = planes * 2
        return nn.Sequential(*layers)
    
    def _add_BasicBlock_layer(self, in_planes, planes, num):
        layers = []
        for i in range(num):
            layers.append(BasicBlock(in_planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x, targets = None):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = F.leaky_relu(self.bn3(self.conv3(out)))
        out = F.leaky_relu(self.bn4(self.conv4(out)))
        out = F.leaky_relu(self.bn5(self.conv5(out)))
        out = F.leaky_relu(self.bn6(self.conv6(out)))
        out = F.leaky_relu(self.bn7(self.conv7(out)))
        out = self.bn8(self.conv8(out))

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.grid_x = torch.arange(self.grid_size).repeat(self.grid_size, 1).view([1, 1, self.grid_size, self.grid_size]).type(FloatTensor)
        self.grid_y = torch.arange(self.grid_size).repeat(self.grid_size, 1).t().view([1, 1, self.grid_size, self.grid_size]).type(FloatTensor)

        num_samples = out.size(0)
        grid_size = out.size(2)

        prediction = (
            out.view(num_samples, 3, 7, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        cx = torch.sigmoid(prediction[..., 0])
        cy = torch.sigmoid(prediction[..., 1])
        dx = torch.tanh(prediction[..., 2])
        dy = torch.tanh(prediction[..., 3])
        w = prediction[..., 4]
        h = prediction[... , 5]
        pred_conf = torch.sigmoid(prediction[..., 6])

        pred_boxes = FloatTensor(prediction[..., :6].shape)
        pred_boxes[..., 0] = cx.data + self.grid_x
        pred_boxes[..., 1] = cy.data + self.grid_y
        pred_boxes[..., 2] = dx.data
        pred_boxes[..., 3] = dy.data
        pred_boxes[..., 4] = w.data
        pred_boxes[..., 5] = h.data

        total_loss = 0

        if targets != None:
            obj_mask, noobj_mask, tcx, tcy, tdx, tdy, tw, th, tconf = build_targets(
            pred_boxes=pred_boxes,
            target=targets  
            )
            
            loss_cx = self.mse_loss(cx[obj_mask], tcx[obj_mask])
            loss_cy = self.mse_loss(cy[obj_mask], tcy[obj_mask])
            loss_dx = self.mse_loss(dx[obj_mask], tdx[obj_mask])
            loss_dy = self.mse_loss(dy[obj_mask], tdy[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            total_loss = loss_cx + loss_cy + loss_dx + loss_dy + loss_w + loss_h + self.obj_scale*loss_conf_obj + self.noobj_scale*loss_conf_noobj
        

        return out, total_loss

#model = PAE()
#print(model)