import glob
import random
import os
import sys
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset

class ListDataset(Dataset):
    def __init__(self, list_path, img_size = 640):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        
        self.label_files = [path.replace("images", "labels").replace(".png",".txt").replace(".jpg",".txt") for path in self.img_files]
        self.img_size = img_size
        self.batch_count = 0
    
    def __getitem__(self, index):
        
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 6))
            targets = torch.zeros((len(boxes), 7))
            targets[:, 1:] = boxes
        
        return img_path, img, targets
    
    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        imgs = torch.stack(imgs)
        self.batch_count += 1
        return paths, imgs, targets
    
    def __len__(self):
        return len(self.img_files)
