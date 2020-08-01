from __future__ import division

import os
import sys
import time
import datetime
import argparse
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from scripts.model import *
from scripts.utils import *
from scripts.dataset import *


def rotate(pt, cx, cy, cs, sn):
    return ((-sn)*(pt[0]-cx)+(-cs)*(pt[1]-cy)+cx, (cs)*(pt[0]-cx)+(-sn)*(pt[1]-cy)+cy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--weights_path", type=str, default="checkpoints/pae_ckpt_74.pth", help="path to weights file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=640, help="size of each image dimension") #416
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = PAE(img_size=opt.img_size).to(device)
    #opt.model_def, 

    model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    #prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        prev_time = time.time()
        # Get detections
        with torch.no_grad():
            detections, _ = model(input_imgs)
            num_samples = detections.size(0)
            grid_size = detections.size(2)
            prediction = (
                detections.view(num_samples, 3, 7, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
            print(prediction.shape)
            for i in range(5):
                for j in range(5):
                    flag = -1
                    max_conf = opt.conf_thres
                    for k in range(3):
                        if torch.sigmoid(prediction[0][k][j][i][6])>max_conf:
                            max_conf = torch.sigmoid(prediction[0][k][j][i][6])
                            flag = k
                    if max_conf > opt.conf_thres:
                        title = "Detection"
                        cx = float(torch.sigmoid(prediction[0][k][j][i][0]).item())*opt.img_size
                        cy = float(torch.sigmoid(prediction[0][k][j][i][1]).item())*opt.img_size
                        dx = float(torch.tanh(prediction[0][k][j][i][2]).item())
                        dy = float(torch.tanh(prediction[0][k][j][i][3]).item())
                        w = float(prediction[0][k][j][i][4].item())*opt.img_size
                        h = float(prediction[0][k][j][i][5].item())*opt.img_size

                        pt1_pred = (cx-w/2, cy-h/2)
                        pt2_pred = (cx-w/2, cy+h/2)
                        pt3_pred = (cx+w/2, cy+h/2)
                        pt4_pred = (cx+w/2, cy-h/2)
                        pt1_pred = rotate(pt1_pred, cx, cy, dx, dy)
                        pt2_pred = rotate(pt2_pred, cx, cy, dx, dy)
                        pt3_pred = rotate(pt3_pred, cx, cy, dx, dy)
                        pt4_pred = rotate(pt4_pred, cx, cy, dx, dy)
                        mx = (pt1_pred[0]+pt4_pred[0])/2
                        my = (pt1_pred[1]+pt4_pred[1])/2
                        pts = [list(pt1_pred), list(pt2_pred), list(pt3_pred), list(pt4_pred), [mx, my], [cx, cy], [mx, my], list(pt1_pred)]
                        xs, ys = zip(*pts)
                        plt.figure()
                        plt.imshow(input_imgs.cpu()[0].permute(1, 2, 0))
                        plt.plot(xs, ys) 
                        plt.show()
                        # pts = np.array([list(pt1_pred), list(pt2_pred), list(pt3_pred), list(pt4_pred), list(pt1_pred)], dtype = np.float32)
                        # pts = pts.reshape((-1, 1, 2))
                        # cv2.polylines(input_imgs.cpu()[0].permute(1, 2, 0).numpy(), np.int32([pts]), False, (0, 0, 255))
                        # cv2.arrowedLine(input_imgs.cpu()[0].permute(1, 2, 0).numpy(), (int(cx), int(cy)), (int(mx), int(my)), (0, 0, 255), 1)
                        # cv2.imshow(title,input_imgs.cpu()[0].permute(1, 2, 0).numpy())

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)