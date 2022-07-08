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

train_transform = get_train_transform()
test_transform = get_val_transform()

net = FasterRCNNDetector(anno_dir=train_anno_path, image_dir=train_image_path, transform = train_transform, test_transform=test_transform, batch_size = 32)
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
        print(image_ids)
        for i, image in enumerate(images):
            boxes = outputs[i]['boxes'].data.cpu().numpy()
            labels = outputs[i]['labels'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()
