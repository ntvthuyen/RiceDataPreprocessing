import sys
sys.path.insert(1,'models')
import pandas as pd
import numpy as np
import cv2
import os

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision
from torchvision.ops import box_iou, MultiScaleRoIAlign
from torchvision import models

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchmetrics import AveragePrecision
from models.faster_rcnn import *
from dataset.rice_dataset import *
from models.model_utils import *

SEED=2484
DEVICE=torch.device('cuda')
pl.utilities.seed.seed_everything(SEED)

train_anno_path = '../../new_annotation/anno_train'
train_image_path = '../../Lua/JPGimages/train'

test_anno_path = '../../new_annotation/anno_test'
test_image_path = '../../Lua/JPGimages/test'

val_anno_path = '../../new_annotation/anno_val'
val_image_path = '../../Lua/JPGimages/valid'

train_transform = get_train_transform()
test_transform = get_val_transform()

net = FasterRCNNDetector(anno_dir=train_anno_path, image_dir=train_image_path, transform = train_transform, test_transform=test_transform, batch_size = 8)
trainer = pl.Trainer(max_epochs=1, gpus=1, progress_bar_refresh_rate=100)
trainer.fit(net)

test_dataset = RiceDataset(test_anno_path, test_image_path, get_val_transform())
test_data_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, pin_memory=True, num_workers=4, collate_fn=collate_fn)

detection_threshold = 0.5
results = []
net.model.to(DEVICE)
net.model.eval()

with torch.no_grad():
    for images, image_ids in test_data_loader:
        images = list(image.to(DEVICE) for image in images)
        outputs = net.model(images)
        #print(image_ids)
        for i, image in enumerate(images):
          image_id = image_ids[i]
          result = {
                  'image_id': image_id,
                  'PredictionString': '14 1.0 0 0 1 1'
                  }
          boxes = outputs[i]['boxes'].data.cpu().numpy()
          labels = outputs[i]['labels'].data.cpu().numpy()
          scores = outputs[i]['scores'].data.cpu().numpy()
          if len(boxes) > 0:
              selected = scores >= detection_threshold
              boxes = boxes[selected].astype(np.int32)
              scores = scores[selected]
              labels = labels[selected]
              if len(boxes) > 0:
                  result = {
                          'image_id': image_id,
                          'PredictionString': format_prediction_string(labels, boxes, scores)
                          }
              results.append(result)
print(results)
