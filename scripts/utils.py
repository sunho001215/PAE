from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def to_cpu(tensor):
    return tensor.detach().cpu()


def build_targets(pred_boxes, target):
    
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nG = pred_boxes.size(2)

    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    tcx = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcy = FloatTensor(nB, nA, nG, nG).fill_(0)
    tdx = FloatTensor(nB, nA, nG, nG).fill_(0)
    tdy = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)

    gcxy = target[:, 1:3] * nG
    gdxy = target[:, 3:5]
    gwh = target[:, 5:7]

    b = target[:,0].long().t()
    gcx, gcy = gcxy.t()
    gdx, gdy = gdxy.t()
    gw, gh = gwh.t()
    gi, gj = gcxy.long().t()

    for i in range(3):
        tcx[b, i, gj, gi] = gcx - gcx.floor()
        tcy[b, i, gj, gi] = gcy - gcy.floor()
        tdx[b, i, gj, gi] = gdx
        tdy[b, i, gj, gi] = gdy
        tw[b, i, gj, gi] = gw
        th[b, i, gj, gi] = gh
        obj_mask[b, i, gj, gi] = 1
        noobj_mask[b, i, gj, gi] = 0

    tconf = obj_mask.float()
    return obj_mask, noobj_mask, tcx, tcy, tdx, tdy, tw, th, tconf