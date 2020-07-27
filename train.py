import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from scripts.model import *
from scripts.dataset import *

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS = 300
BATCH_SIZE = 128

model = PAE.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr= 0.1, momentum= 0.9, weight_decay = 0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 50, gamma= 0.7)
