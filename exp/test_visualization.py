from pprint import pprint
from models.model_utils import *
from dataset.rice_dataset import *
from models.faster_rcnn import *
from torchmetrics import AveragePrecision
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import models
from torchvision.ops import box_iou, MultiScaleRoIAlign
import torchvision
import torch
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import os
import cv2
import numpy as np
import pandas as pd
import sys

sys.path.insert(1, 'models')
from utils.visualization import *
SEED = 2484
DEVICE = torch.device('cuda')
pl.utilities.seed.seed_everything(SEED)

train_anno_path = '../../new_annotation/anno_train'
train_image_path = '../../Lua/JPGimages/train'

test_anno_path = '../../new_annotation/anno_test'
test_image_path = '../../Lua/JPGimages/test'

val_anno_path = '../../new_annotation/anno_val'
val_image_path = '../../Lua/JPGimages/valid'

train_transform = get_train_transform()
test_transform = get_val_transform()

net = FasterRCNNDetector.load_from_checkpoint('checkpoint.ckpt', anno_dir=train_anno_path,
                                              image_dir=train_image_path, transform=train_transform, test_transform=test_transform, batch_size=16)

test_dataset = RiceDataset(
    test_anno_path, test_image_path, get_val_transform())
test_data_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,
                              pin_memory=True, num_workers=4, collate_fn=collate_fn)



detection_threshold = 0.5
results = []
net.model.to(DEVICE)
net.model.eval()

prediction_l1 = []
ground_truth_l1 = []

prediction_l2 = []
ground_truth_l2 = []

prediction_l3 = []
ground_truth_l3 = []

prediction = []
ground_truth = []


test_images = sorted(os.listdir(test_image_path))
t = 0

image_transform = A.Compose([
        A.LongestMaxSize(max_size=1024)
    ])

with torch.no_grad():
    for images, targets in test_data_loader:
        images = list(image.to(DEVICE) for image in images)
        outputs = net.model(images)

        for i, image in enumerate(images):
            target = targets[i]
            ground_truth.append(dict(
                boxes=target['boxes'].to(DEVICE),
                labels=target['labels'].to(DEVICE))
            )

            boxes = outputs[i]['boxes']
            labels = outputs[i]['labels']
            scores = outputs[i]['scores']
            if len(boxes) > 0:
                selected = scores >= detection_threshold

                prediction.append(dict(
                    boxes=boxes[selected],
                    scores=scores[selected],
                    labels=labels[selected]
                ))
            else:
                prediction.append(dict(
                    boxes=torch.tensor([], device=DEVICE),
                    scores=torch.tensor([], device=DEVICE),
                    labels=torch.tensor([], device=DEVICE)
                ))
            
            if target['labels'][0] == 1.0:
                ground_truth_l1.append(ground_truth[-1])        
            elif target['labels'][0] == 2.0:
                ground_truth_l1.append(ground_truth[-1])        
            else:
                ground_truth_l3.append(ground_truth[-1])        

            if target['labels'][0] == 1.0:
                prediction_l1.append(prediction[-1])        
            elif target['labels'][0] == 2.0:
                prediction_l2.append(prediction[-1])        
            else:
                prediction_l3.append(prediction[-1])        
            
            if test_transform:
                timage = image_transform(image=cv2.imread(test_image_path+'/'+test_images[t]))
            else:
                timage = cv2.imread(test_image_path+'/'+test_images[t])
            visualize_single_image(timage['image'], prediction[-1]['boxes'].data.cpu().numpy(),ground_truth[len(prediction)-1]['boxes'].data.cpu().numpy(),'test_visualization/' +test_images[t])
            t=t+1
print(len(prediction), len(ground_truth))
metric = MeanAveragePrecision()
metric.update(prediction, ground_truth)
pprint(metric.compute())

print(len(prediction_l1), len(ground_truth_l1))
metric = MeanAveragePrecision()
metric.update(prediction_l1, ground_truth_l1)
pprint(metric.compute())

print(len(prediction_l2), len(ground_truth_l2))
metric = MeanAveragePrecision()
metric.update(prediction_l2, ground_truth_l2)
pprint(metric.compute())

print(len(prediction_l3), len(ground_truth_l3))
metric = MeanAveragePrecision()
metric.update(prediction_l3, ground_truth_l3)
pprint(metric.compute())
